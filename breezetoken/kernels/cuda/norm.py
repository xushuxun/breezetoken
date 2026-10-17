"""CUDA f32 RMSNorm and SwiGLU helpers.

Activations stay f32 on the GPU (only the projection weights are bf16), so
these kernels use f32 throughout.  That keeps the arithmetic identical to the
torch reference and avoids mixed-dtype buffer plumbing.
"""

from __future__ import annotations

from functools import cache

import tilelang
import tilelang.language as T

DT = "float32"


@cache
def rms_norm(M: int, N: int, eps: float, threads: int = 256):
    """One block per row; ``M`` rows of length ``N``."""

    @T.prim_func
    def main(X: T.Tensor((M, N), DT), G: T.Tensor((N,), DT), O: T.Tensor((M, N), DT)):
        with T.Kernel(M, threads=threads) as row:
            xs = T.alloc_fragment((N,), DT)
            sq = T.alloc_fragment((N,), DT)
            total = T.alloc_fragment((1,), DT)
            for i in T.Parallel(N):
                xs[i] = X[row, i]
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, total, dim=0)
            for i in T.Parallel(N):
                O[row, i] = xs[i] * T.rsqrt(total[0] / N + eps) * G[i]

    return tilelang.compile(main)


@cache
def silu_mul(I: int, block: int = 256, threads: int = 256):
    """``O = silu(G) * U``."""

    @T.prim_func
    def main(G: T.Tensor((I,), DT), U: T.Tensor((I,), DT), O: T.Tensor((I,), DT)):
        with T.Kernel(T.ceildiv(I, block), threads=threads) as bx:
            for j in T.Parallel(block):
                idx = bx * block + j
                with T.If(idx < I), T.Then():
                    g = T.max(G[idx], -80.0)
                    O[idx] = (g / (1.0 + T.exp(-g))) * U[idx]

    return tilelang.compile(main)
