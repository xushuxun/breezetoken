"""TileLang CPU-target kernels for the Qwen3-0.6B decode step.

Compiled with tilelang target="c" (the LLVM path is unavailable in this
install's TVM build). Consequences, measured on this machine:

  * dtype is float32 end-to end -- the C codegen cannot emit bfloat16
  * generated code is scalar and single-threaded (~1 GFLOPS ceiling);
    the CPU backend here is a correctness-oriented reference backend

Kernel contracts mirror kernels.py (GPU): every kernel takes caller-owned
output buffers and a device ``Pos`` tensor, so the step is a straight
per-token loop. No split-K: GEMVs write their output directly.
"""
from functools import lru_cache

import tilelang
import tilelang.language as T

DT = "float32"

# cc flags for the generated C: without these the backend emits scalar code at
# -O0-class performance (~0.8 GFLOPS); with them gcc auto-vectorizes and the
# GEMVs become single-core-bandwidth-bound (~8 GFLOPS)
CC_FLAGS = ["-O3", "-march=native", "-ffast-math"]


def _exp(x):
    # the C codegen resolves only a handful of math intrinsics (rsqrt); exp and
    # friends are emitted verbatim via call_extern and resolved by libm
    return T.call_extern("float32", "expf", x)


@lru_cache(maxsize=None)
def embed(V: int, H: int):
    @T.prim_func
    def main(Tbl: T.Tensor((V, H), DT), Ids: T.Tensor((1,), "int64"),
             O: T.Tensor((H,), DT)):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(H):
                O[i] = Tbl[Ids[0], i]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def rms_norm(H: int, eps: float):
    @T.prim_func
    def main(X: T.Tensor((H,), DT), G: T.Tensor((H,), DT), O: T.Tensor((H,), DT)):
        with T.Kernel(1, threads=1):
            tot = T.alloc_fragment((1,), DT)
            tot[0] = 0.0
            for i in T.serial(H):
                tot[0] += X[i] * X[i]
            for i in T.Parallel(H):
                O[i] = X[i] * T.rsqrt(tot[0] / H + eps) * G[i]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def resid_rms_norm(H: int, eps: float):
    """Hout = A + B; Xn = rms_norm(Hout) * G -- the residual add and the norm
    that follows it are one read of the same vector."""
    @T.prim_func
    def main(A: T.Tensor((H,), DT), B: T.Tensor((H,), DT), G: T.Tensor((H,), DT),
             Hout: T.Tensor((H,), DT), Xn: T.Tensor((H,), DT)):
        with T.Kernel(1, threads=1):
            tot = T.alloc_fragment((1,), DT)
            tot[0] = 0.0
            for i in T.serial(H):
                v = A[i] + B[i]
                Hout[i] = v
                tot[0] += v * v
            for i in T.Parallel(H):
                Xn[i] = Hout[i] * T.rsqrt(tot[0] / H + eps) * G[i]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def gemv(K: int, N: int, BK: int = 64, BN: int = 64):
    """O = X @ W, W stored (K, N) row-major. Blocked over K so each W tile is a
    contiguous unit-stride copy into a local buffer."""
    @T.prim_func
    def main(X: T.Tensor((K,), DT), W: T.Tensor((K, N), DT), O: T.Tensor((N,), DT)):
        with T.Kernel(T.ceildiv(N, BN), threads=1) as bx:
            Wt = T.alloc_fragment((BK, BN), DT)
            Xt = T.alloc_fragment((BK,), DT)
            acc = T.alloc_fragment((BN,), DT)
            for j in T.Parallel(BN):
                acc[j] = 0.0
            for ko in T.serial(K // BK):
                T.copy(W[ko * BK:(ko + 1) * BK, bx * BN:(bx + 1) * BN], Wt)
                T.copy(X[ko * BK:(ko + 1) * BK], Xt)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += Xt[kk] * Wt[kk, j]
            for j in T.Parallel(BN):
                O[bx * BN + j] = acc[j]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def qk_rope_cache(HQ: int, HKV: int, D: int, CAP: int, eps: float):
    """Per-head q/k RMSNorm + rotate-half RoPE + cache write, from the fused QKV vector.

    X is [q | k | v] over the flat axis; query heads land in Q, this step's
    key/value go into Kc/Vc at Pos. Value heads pass through un-normed.
    """
    QN, KN = HQ * D, HKV * D
    H = D // 2

    @T.prim_func
    def main(X: T.Tensor((QN + 2 * KN,), DT),
             Gq: T.Tensor((D,), DT), Gk: T.Tensor((D,), DT),
             Cos: T.Tensor((CAP, D), DT), Sin: T.Tensor((CAP, D), DT),
             Pos: T.Tensor((1,), "int32"),
             Kc: T.Tensor((CAP, KN), DT), Vc: T.Tensor((CAP, KN), DT),
             Q: T.Tensor((QN,), DT)):
        with T.Kernel(HQ + 2 * HKV, threads=1) as bh:
            is_q = bh < HQ
            is_k = bh < HQ + HKV
            head = T.if_then_else(is_q, bh, T.if_then_else(is_k, bh - HQ, bh - HQ - HKV))
            base = T.if_then_else(is_q, 0, T.if_then_else(is_k, QN, QN + KN)) + head * D
            tot = T.alloc_fragment((1,), DT)
            tot[0] = 0.0
            for d in T.serial(D):
                tot[0] += X[base + d] * X[base + d]
            scale = T.rsqrt(tot[0] / D + eps)
            for d in T.Parallel(D):
                gv = T.if_then_else(is_q, Gq[d], Gk[d])
                xn = X[base + d] * scale * gv
                p = T.if_then_else(d < H, d + H, d - H)
                gp = T.if_then_else(is_q, Gq[p], Gk[p])
                xn_p = X[base + p] * scale * gp
                sgn = T.if_then_else(d < H, -1.0, 1.0)
                rot = xn * Cos[Pos[0], d] + sgn * xn_p * Sin[Pos[0], d]
                with T.If(is_q):
                    with T.Then():
                        Q[head * D + d] = rot
                    with T.Else():
                        with T.If(is_k):
                            with T.Then():
                                Kc[Pos[0], head * D + d] = rot
                            with T.Else():
                                Vc[Pos[0], head * D + d] = X[base + d]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def attn_decode(HQ: int, HKV: int, D: int, CAP: int, scale: float):
    """One query row per head against Kc/Vc[0:Pos+1]; GQA group per kv head.

    Per kv head: scores for each of its G query heads in a (G, CAP) fragment,
    masked softmax, then the weighted sum over V. Serial throughout -- the CPU
    backend lowers each block to a single thread."""
    G = HQ // HKV

    @T.prim_func
    def main(Q: T.Tensor((HQ * D,), DT),
             Kc: T.Tensor((CAP, HKV * D), DT), Vc: T.Tensor((CAP, HKV * D), DT),
             Pos: T.Tensor((1,), "int32"), O: T.Tensor((HQ * D,), DT)):
        with T.Kernel(HKV, threads=1) as bh:
            sc = T.alloc_fragment((G, CAP), DT)
            mx = T.alloc_fragment((G,), DT)
            den = T.alloc_fragment((G,), DT)
            acc = T.alloc_fragment((1,), DT)
            for g in T.serial(G):
                mx[g] = -1.0e30
                for s in T.serial(CAP):
                    acc[0] = 0.0
                    for d in T.serial(D):
                        acc[0] += Q[(bh * G + g) * D + d] * Kc[s, bh * D + d]
                    acc[0] = T.if_then_else(s <= Pos[0], acc[0] * scale, -1.0e30)
                    sc[g, s] = acc[0]
                    mx[g] = T.max(mx[g], acc[0])
                den[g] = 0.0
                for s in T.serial(CAP):
                    sc[g, s] = _exp(sc[g, s] - mx[g])
                    den[g] += sc[g, s]
                for d in T.serial(D):
                    acc[0] = 0.0
                    for s in T.serial(CAP):
                        acc[0] += sc[g, s] * Vc[s, bh * D + d]
                    O[(bh * G + g) * D + d] = acc[0] / den[g]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)


@lru_cache(maxsize=None)
def silu_mul(I: int):
    """O = silu(X[:I]) * X[I:2I]."""
    @T.prim_func
    def main(X: T.Tensor((2 * I,), DT), O: T.Tensor((I,), DT)):
        with T.Kernel(T.ceildiv(I, 256), threads=1) as bx:
            for j in T.Parallel(256):
                idx = bx * 256 + j
                with T.If(idx < I):
                    with T.Then():
                        # clamp before expf: with -ffast-math the compiler may
                        # reassociate the division so expf overflow turns into
                        # inf*0 = NaN; silu(g) for g < -80 is ~0 either way
                        g = T.max(X[idx], -80.0)
                        O[idx] = (g / (1.0 + _exp(-g))) * X[I + idx]
    return tilelang.compile(main, target="c", compile_flags=CC_FLAGS)
