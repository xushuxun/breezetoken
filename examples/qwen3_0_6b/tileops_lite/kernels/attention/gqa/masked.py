"""Graph-safe GQA decode attention: fixed-capacity KV with device-side Pos mask.

Derived from TileOPs' ``gqa_decode_no_split_kernel`` (MIT, see TileOPs LICENSE)
with two changes that make the whole decode step CUDA-graph capturable:

  * K/V are the full fixed-capacity cache (static shapes, static trip count)
    instead of a dynamically-shaped exact-length view
  * the live length is read from a device ``Pos`` tensor and scores beyond it
    are masked to -inf -- the same contract as the TileFoundry attn_partial
"""
import functools

import tilelang
import tilelang.language as T

from tileops_lite.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops_lite.kernels.constants import LOG2E


@functools.lru_cache(maxsize=None)
def gqa_decode_masked(batch: int, heads: int, groups: int, dim: int, cap: int,
                      sm_scale, dtype: str, block_N: int = 64, threads: int = 128):
    score_scale = dim ** -0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups
    block_H = 64  # upstream default; smaller M hits a layout-inference conflict
                  # on the f32->bf16 score cast

    valid_block_H = min(block_H, kv_group_num)
    blocks_per_kv = -(-kv_group_num // valid_block_H)
    tail_rows = kv_group_num - (blocks_per_kv - 1) * valid_block_H

    # fully-masked tiles exist by construction (past Pos): the guard
    # variant keeps exp2(-inf - -inf) from ever producing NaN
    online_softmax = make_online_softmax_with_mask_guard(
        scale, accum_dtype, block_H, block_N)
    rescale = make_rescale(block_H, dim)

    shape_q = [batch, heads, dim]
    shape_kv = [batch, cap, groups, dim]
    shape_o = [batch, heads, dim]

    @T.macro
    def compute(Q, K, V, Pos, Output):
        with T.Kernel(batch, groups * blocks_per_kv, 1, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // blocks_per_kv
            head_base = cur_kv_head * kv_group_num + hid % blocks_per_kv * valid_block_H
            rows = (
                valid_block_H
                if tail_rows == valid_block_H
                else T.if_then_else(
                    hid % blocks_per_kv == blocks_per_kv - 1, tail_rows, valid_block_H
                )
            )

            if tail_rows == valid_block_H:
                T.copy(Q[bid, head_base : head_base + block_H, :], Q_shared)
            else:
                for i, j in T.Parallel(block_H, dim):
                    if i < rows:
                        Q_shared[i, j] = Q[bid, head_base + i, j]
                    else:
                        Q_shared[i, j] = 0
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(cap // block_N, num_stages=1):
                T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(
                    Q_shared, K_shared, acc_s, transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j < Pos[bid] + 1,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )
                online_softmax(
                    acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                )
                T.copy(acc_s, acc_s_cast)
                rescale(acc_o, scores_scale)
                T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o[:valid_block_H, :], O_shared)
            if tail_rows == valid_block_H:
                T.copy(O_shared, Output[bid, head_base : head_base + valid_block_H, :])
            else:
                for i, j in T.Parallel(valid_block_H, dim):
                    if i < rows:
                        Output[bid, head_base + i, j] = O_shared[i, j]

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Pos: T.Tensor((batch,), "int32"),
        Output: T.Tensor(shape_o, dtype),
    ):
        compute(Q, K, V, Pos, Output)

    return tilelang.compile(main)
