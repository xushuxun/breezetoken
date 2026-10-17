"""CPU GEMV: ``O = X @ W`` with W row-major ``(K, N)``."""

from __future__ import annotations

from functools import cache

import tilelang
import tilelang.language as T

from breezetoken.kernels.cpu.common import CC_FLAGS, DT, TARGET


@cache
def gemv(K: int, N: int, BK: int = 64, BN: int = 64):
    """Blocked over K so each W tile is a contiguous unit-stride copy.

    Callers must validate ``K % BK == 0`` and ``N % BN == 0``; the backend
    does that before selecting this kernel.
    """

    @T.prim_func
    def main(X: T.Tensor((K,), DT), W: T.Tensor((K, N), DT), O: T.Tensor((N,), DT)):
        with T.Kernel(T.ceildiv(N, BN), threads=1) as bx:
            wt = T.alloc_fragment((BK, BN), DT)
            xt = T.alloc_fragment((BK,), DT)
            acc = T.alloc_fragment((BN,), DT)
            for j in T.Parallel(BN):
                acc[j] = 0.0
            for ko in T.serial(K // BK):
                T.copy(W[ko * BK : (ko + 1) * BK, bx * BN : (bx + 1) * BN], wt)
                T.copy(X[ko * BK : (ko + 1) * BK], xt)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += xt[kk] * wt[kk, j]
            for j in T.Parallel(BN):
                O[bx * BN + j] = acc[j]

    return tilelang.compile(main, target=TARGET, compile_flags=CC_FLAGS)
