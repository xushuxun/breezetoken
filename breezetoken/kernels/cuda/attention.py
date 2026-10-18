"""GQA decode attention: context-split flash kernel + log-sum-exp combine.

Adapted from TileOPs' GQA decode split/combine kernels (MIT; see NOTICE).
The 0.8B shape is ``heads=8, kv_heads=2, dim=256``.  A 99KB shared-memory
budget cannot hold a 64-row query tile, so the query tile is 16 rows (4 valid
rows padded to 16 for MMA) and K/V tiles are bf16.  Context is split into
``num_split`` blocks; each writes a partial output and log-sum-exp, and a
second in-module kernel combines them.

The caller owns the KV cache; the kernel only reads the current prefix.
"""

from functools import cache

import tilelang
import tilelang.language as T

LOG2E = 1.4426950408889634


def _online_softmax(scale, block_rows, block_cols):
    @T.macro
    def online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity("float"))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_rows):
            # A later fully-masked tile must not erase the running maximum;
            # lse is computed from scores_max after the loop.
            scores_max[i] = T.if_then_else(
                scores_max[i] > -T.infinity("float"), scores_max[i], scores_max_prev[i]
            )
        for i in T.Parallel(block_rows):
            # A fully-masked tile leaves both maxima at -inf; keep the running
            # accumulator unchanged instead of evaluating (-inf) - (-inf).
            scores_scale[i] = T.if_then_else(
                scores_max[i] > -T.infinity("float"),
                T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                1.0,
            )
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.if_then_else(
                scores_max[i] > -T.infinity("float"),
                T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                0.0,
            )
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_rows):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    return online_softmax


@cache
def gqa_decode(
    heads: int,
    groups: int,
    dim: int,
    scale: float,
    num_split: int = 16,
    block_h: int = 16,
    block_n: int = 32,
    threads: int = 128,
    dtype: str = "float32",
):
    """Build the split attention kernel for one batch item."""
    if heads % groups:
        raise ValueError("heads must be divisible by groups")
    if heads // groups > block_h:
        raise ValueError("the query group must fit in one padded query tile")
    accum = "float"
    softmax_scale = scale * LOG2E
    group_size = heads // groups
    online_softmax = _online_softmax(softmax_scale, block_h, block_n)

    @tilelang.jit(
        compile_flags=["-O3"],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _func():
        seqlen = T.dynamic("seqlen")
        q_shape = (1, heads, dim)
        kv_shape = (1, seqlen, groups, dim)
        o_shape = (1, heads, dim)
        lse_shape = (1, heads, num_split)
        part_shape = (1, heads, num_split, dim)

        @T.macro
        def split(Q, K, V, Lse, Part):
            with T.Kernel(1, groups, num_split, threads=threads) as (_, by, bz):
                q_shared = T.alloc_shared([block_h, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                p_shared = T.alloc_shared([block_h, block_n], accum)
                o_shared = T.alloc_shared([group_size, dim], dtype)
                acc_s = T.alloc_fragment([block_h, block_n], accum)
                acc_o = T.alloc_fragment([block_h, dim], accum)
                scores_max = T.alloc_fragment([block_h], accum)
                scores_max_prev = T.alloc_fragment([block_h], accum)
                scores_scale = T.alloc_fragment([block_h], accum)
                scores_sum = T.alloc_fragment([block_h], accum)
                logsum = T.alloc_fragment([block_h], accum)

                kv_head = by
                head_base = kv_head * group_size
                num_tiles = T.ceildiv(seqlen, block_n)
                base_tiles = num_tiles // num_split
                extra = num_tiles % num_split
                tiles = base_tiles + T.if_then_else(bz < extra, 1, 0)
                tile_begin = bz * base_tiles + T.min(bz, extra)
                base = tile_begin * block_n
                this_len = T.max(T.min(seqlen - base, tiles * block_n), 0)

                for i, j in T.Parallel(block_h, dim):
                    if i < group_size:
                        q_shared[i, j] = Q[0, head_base + i, j]
                    else:
                        q_shared[i, j] = 0

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum))

                for k in T.Pipelined(T.ceildiv(this_len, block_n), num_stages=1):
                    T.copy(
                        K[0, base + k * block_n : base + (k + 1) * block_n, kv_head, :], k_shared
                    )
                    T.clear(acc_s)
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(block_h, block_n):
                        acc_s[i, j] = T.if_then_else(
                            base + k * block_n + j < seqlen,
                            acc_s[i, j],
                            -T.infinity(accum),
                        )
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, p_shared)
                    for i, j in T.Parallel(block_h, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[0, base + k * block_n : base + (k + 1) * block_n, kv_head, :], v_shared
                    )
                    T.gemm(p_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_h, dim):
                    acc_o[i, j] /= T.if_then_else(this_len > 0, logsum[i], 1.0)
                for i in T.Parallel(block_h):
                    logsum[i] = T.if_then_else(
                        this_len > 0,
                        T.log2(logsum[i]) + scores_max[i] * softmax_scale,
                        -T.infinity(accum),
                    )
                for i in T.Parallel(block_h):
                    if i < group_size:
                        Lse[0, head_base + i, bz] = logsum[i]
                for i, j in T.Parallel(group_size, dim):
                    o_shared[i, j] = acc_o[i, j]
                T.copy(o_shared, Part[0, head_base : head_base + group_size, bz, :])

        @T.macro
        def combine(Lse, Part, Out):
            with T.Kernel(heads, 1, threads=128) as (by, _):
                lse = T.alloc_fragment([num_split], accum)
                for k in T.Parallel(num_split):
                    lse[k] = Lse[0, by, k]
                lse_max = T.alloc_fragment([1], accum)
                T.fill(lse_max, -T.infinity(accum))
                T.reduce_max(lse, lse_max, dim=0, clear=False)
                total = T.alloc_local([1], accum)
                total[0] = 0
                for k in T.serial(num_split):
                    total[0] += T.exp2(Lse[0, by, k] - lse_max[0])
                total[0] = T.log2(total[0]) + lse_max[0]
                out = T.alloc_fragment([dim], accum)
                T.clear(out)
                for k in T.serial(num_split):
                    weight = T.exp2(Lse[0, by, k] - total[0])
                    for i in T.Parallel(dim):
                        out[i] += Part[0, by, k, i] * weight
                for i in T.Parallel(dim):
                    Out[0, by, i] = out[i]

        @T.prim_func
        def kernel(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Lse: T.Tensor(lse_shape, accum),
            Part: T.Tensor(part_shape, dtype),
            Out: T.Tensor(o_shape, dtype),
        ):
            split(Q, K, V, Lse, Part)
            combine(Lse, Part, Out)

        return kernel

    return _func()


@cache
def gqa_decode_fixed(
    heads: int,
    groups: int,
    dim: int,
    cap: int,
    scale: float,
    num_split: int = 16,
    block_h: int = 16,
    block_n: int = 32,
    threads: int = 128,
    dtype: str = "float32",
):
    """Fixed-capacity variant used by the CUDA graph decode path.

    K/V are the full preallocated cache and ``pos`` is a device int32.  The
    kernel masks positions beyond ``pos``; the split partitions are fixed, so
    no tensor shapes change between replay steps.
    """
    if cap % block_n:
        raise ValueError("cap must be a multiple of the K/V tile")
    if heads % groups:
        raise ValueError("heads must be divisible by groups")
    if heads // groups > block_h:
        raise ValueError("the query group must fit in one padded query tile")
    accum = "float"
    softmax_scale = scale * LOG2E
    group_size = heads // groups
    online_softmax = _online_softmax(softmax_scale, block_h, block_n)

    @tilelang.jit(
        compile_flags=["-O3"],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _func():
        q_shape = (1, heads, dim)
        kv_shape = (1, cap, groups, dim)
        o_shape = (1, heads, dim)
        lse_shape = (1, heads, num_split)
        part_shape = (1, heads, num_split, dim)

        @T.macro
        def split(Q, K, V, Pos, Lse, Part):
            with T.Kernel(1, groups, num_split, threads=threads) as (_, by, bz):
                q_shared = T.alloc_shared([block_h, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                p_shared = T.alloc_shared([block_h, block_n], accum)
                o_shared = T.alloc_shared([group_size, dim], dtype)
                acc_s = T.alloc_fragment([block_h, block_n], accum)
                acc_o = T.alloc_fragment([block_h, dim], accum)
                scores_max = T.alloc_fragment([block_h], accum)
                scores_max_prev = T.alloc_fragment([block_h], accum)
                scores_scale = T.alloc_fragment([block_h], accum)
                scores_sum = T.alloc_fragment([block_h], accum)
                logsum = T.alloc_fragment([block_h], accum)

                kv_head = by
                head_base = kv_head * group_size
                num_tiles = cap // block_n
                base_tiles = num_tiles // num_split
                extra = num_tiles % num_split
                tiles = base_tiles + T.if_then_else(bz < extra, 1, 0)
                tile_begin = bz * base_tiles + T.min(bz, extra)
                base = tile_begin * block_n

                for i, j in T.Parallel(block_h, dim):
                    if i < group_size:
                        q_shared[i, j] = Q[0, head_base + i, j]
                    else:
                        q_shared[i, j] = 0

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum))

                for k in T.Pipelined(tiles, num_stages=1):
                    T.copy(
                        K[0, base + k * block_n : base + (k + 1) * block_n, kv_head, :], k_shared
                    )
                    T.clear(acc_s)
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(block_h, block_n):
                        acc_s[i, j] = T.if_then_else(
                            base + k * block_n + j <= Pos[0],
                            acc_s[i, j],
                            -T.infinity(accum),
                        )
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, p_shared)
                    for i, j in T.Parallel(block_h, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[0, base + k * block_n : base + (k + 1) * block_n, kv_head, :], v_shared
                    )
                    T.gemm(p_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_h, dim):
                    acc_o[i, j] /= T.if_then_else(logsum[i] > 0, logsum[i], 1.0)
                for i in T.Parallel(block_h):
                    logsum[i] = T.if_then_else(
                        logsum[i] > 0,
                        T.log2(logsum[i]) + scores_max[i] * softmax_scale,
                        -T.infinity(accum),
                    )
                for i in T.Parallel(block_h):
                    if i < group_size:
                        Lse[0, head_base + i, bz] = logsum[i]
                for i, j in T.Parallel(group_size, dim):
                    o_shared[i, j] = acc_o[i, j]
                T.copy(o_shared, Part[0, head_base : head_base + group_size, bz, :])

        @T.macro
        def combine(Lse, Part, Out):
            with T.Kernel(heads, 1, threads=128) as (by, _):
                lse = T.alloc_fragment([num_split], accum)
                for k in T.Parallel(num_split):
                    lse[k] = Lse[0, by, k]
                lse_max = T.alloc_fragment([1], accum)
                T.fill(lse_max, -T.infinity(accum))
                T.reduce_max(lse, lse_max, dim=0, clear=False)
                total = T.alloc_local([1], accum)
                total[0] = 0
                for k in T.serial(num_split):
                    total[0] += T.exp2(Lse[0, by, k] - lse_max[0])
                total[0] = T.log2(total[0]) + lse_max[0]
                out = T.alloc_fragment([dim], accum)
                T.clear(out)
                for k in T.serial(num_split):
                    weight = T.exp2(Lse[0, by, k] - total[0])
                    for i in T.Parallel(dim):
                        out[i] += Part[0, by, k, i] * weight
                for i in T.Parallel(dim):
                    Out[0, by, i] = out[i]

        @T.prim_func
        def kernel(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Pos: T.Tensor((1,), "int32"),
            Lse: T.Tensor(lse_shape, accum),
            Part: T.Tensor(part_shape, dtype),
            Out: T.Tensor(o_shape, dtype),
        ):
            split(Q, K, V, Pos, Lse, Part)
            combine(Lse, Part, Out)

        return kernel

    return _func()
