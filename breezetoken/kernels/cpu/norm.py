"""CPU RMSNorm and SwiGLU helpers (f32, one row per block)."""

from functools import cache

import tilelang
import tilelang.language as T

from breezetoken.kernels.cpu.common import CC_FLAGS, DT, TARGET, _exp


@cache
def rms_norm(M: int, N: int, eps: float):
    """``O[r, :] = X[r, :] * rsqrt(mean(X[r, :]^2) + eps) * G``.

    ``M`` rows are independent; the model uses this for flat hidden states,
    per-head q/k norms, and the GDN output norm.
    """

    @T.prim_func
    def main(X: T.Tensor((M, N), DT), G: T.Tensor((N,), DT), O: T.Tensor((M, N), DT)):
        with T.Kernel(M, threads=1) as row:
            total = T.alloc_fragment((1,), DT)
            total[0] = 0.0
            for i in T.serial(N):
                total[0] += X[row, i] * X[row, i]
            for i in T.Parallel(N):
                O[row, i] = X[row, i] * T.rsqrt(total[0] / N + eps) * G[i]

    return tilelang.compile(main, target=TARGET, compile_flags=CC_FLAGS)


@cache
def silu_mul(I: int, block: int = 256):
    """``O = silu(G) * U`` over one flat vector."""

    @T.prim_func
    def main(G: T.Tensor((I,), DT), U: T.Tensor((I,), DT), O: T.Tensor((I,), DT)):
        with T.Kernel(T.ceildiv(I, block), threads=1) as bx:
            for j in T.Parallel(block):
                idx = bx * block + j
                with T.If(idx < I), T.Then():
                    # clamp before expf: -ffast-math may turn exp overflow
                    # into inf * 0 = NaN. silu(g) is ~0 below -80 anyway.
                    g = T.max(G[idx], -80.0)
                    O[idx] = (g / (1.0 + _exp(-g))) * U[idx]

    return tilelang.compile(main, target=TARGET, compile_flags=CC_FLAGS)


@cache
def resid_rms_norm(M: int, N: int, eps: float):
    """``Hout = A + B`` followed by RMSNorm(Hout); one kernel and one read."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), DT),
        B: T.Tensor((M, N), DT),
        G: T.Tensor((N,), DT),
        Hout: T.Tensor((M, N), DT),
        Xn: T.Tensor((M, N), DT),
    ):
        with T.Kernel(M, threads=1) as row:
            total = T.alloc_fragment((1,), DT)
            total[0] = 0.0
            for i in T.serial(N):
                v = A[row, i] + B[row, i]
                Hout[row, i] = v
                total[0] += v * v
            for i in T.Parallel(N):
                Xn[row, i] = Hout[row, i] * T.rsqrt(total[0] / N + eps) * G[i]

    return tilelang.compile(main, target=TARGET, compile_flags=CC_FLAGS)
