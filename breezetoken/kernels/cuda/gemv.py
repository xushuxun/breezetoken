"""CUDA bf16 decode GEMV.

Ported from TileFoundry (MIT); see ``NOTICE`` and
``examples/qwen3_0_6b/LICENSE.tilefoundry``.  Tile shapes respect the target
GPU's shared-memory opt-in ceiling; the backend checks them before selecting
this kernel.
"""

from __future__ import annotations

from functools import cache

import tilelang
import tilelang.language as T

DT = "bfloat16"
ACC = "float32"


@cache
def gemv(K: int, N: int, BN: int, BK: int, SK: int, threads: int, stages: int = 3):
    """``P[s, n] = sum(X[k] * W[k, n])`` over split *s*'s share of ``K``."""
    KS = K // SK

    @T.prim_func
    def main(X: T.Tensor((K,), DT), W: T.Tensor((K, N), DT), P: T.Tensor((SK, N), ACC)):
        with T.Kernel(T.ceildiv(N, BN), SK, threads=threads) as (bx, bs):
            ws = T.alloc_shared((BK, BN), DT)
            xs = T.alloc_shared((BK,), DT)
            acc = T.alloc_fragment((BN,), ACC)
            T.clear(acc)
            for ko in T.Pipelined(KS // BK, num_stages=stages):
                k0 = bs * KS + ko * BK
                T.copy(W[k0 : k0 + BK, bx * BN : (bx + 1) * BN], ws)
                T.copy(X[k0 : k0 + BK], xs)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += xs[kk].astype(ACC) * ws[kk, j].astype(ACC)
            for j in T.Parallel(BN):
                P[bs, bx * BN + j] = acc[j]

    return tilelang.compile(main)
