"""Single-token Gated DeltaNet decode kernel.

Adapted from TileOPs' ``delta_decode`` (MIT, ``TileOPs`` checkout in this
workspace; see ``NOTICE`` and ``LICENSES/TileOPs-MIT.txt``).  The original
advances one value-column tile of the recurrent state per block and returns a
new state; this version updates the caller's f32 state in place, which is how
the decode loop wants it.

One token, one f32 state ``[heads, K, V]`` with ``K == V`` in (64, 128).
Each block owns 32 value columns of one head; a lane group of two splits the
key dimension.  The query/key L2 normalization, the beta sigmoid and the gated
decay are fused here, so the model only computes the projections around it.
"""

from functools import cache

import tilelang
import tilelang.language as T

LOG2E = 1.4426950408889634
WARP_LANES = 32
WARP_SHUFFLE_STAGES = 5


@cache
def gdn_decode(
    heads: int,
    dim: int,
    scale: float,
    threads: int = 64,
    lane_group: int = 2,
):
    """Build the decode kernel for ``[heads, dim, dim]`` f32 state."""
    if dim not in (64, 128):
        raise ValueError(f"GDN decode requires K == V in (64, 128), got {dim}")
    if threads % WARP_LANES or lane_group % 1 or WARP_LANES % lane_group:
        raise ValueError("threads must be whole warps and lane_group must divide 32")
    if dim % lane_group:
        raise ValueError("dim must be divisible by lane_group")
    value_tile = threads // lane_group
    if dim % value_tile:
        raise ValueError(f"dim must be divisible by the {value_tile} columns per block")
    value_tiles = dim // value_tile
    total_blocks = heads * value_tiles
    k_chunk = dim // lane_group
    lane_group_stages = lane_group.bit_length() - 1
    l2_eps = 1e-6

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3"])
    def _decode():
        @T.prim_func
        def kernel(
            q: T.Tensor((1, 1, heads, dim), "float32"),
            k: T.Tensor((1, 1, heads, dim), "float32"),
            v: T.Tensor((1, 1, heads, dim), "float32"),
            g: T.Tensor((1, 1, heads), "float32"),
            beta: T.Tensor((1, 1, heads), "float32"),
            state: T.Tensor((1, heads, dim, dim), "float32"),
            o: T.Tensor((1, 1, heads, dim), "float32"),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                value_tile_idx = block % value_tiles
                head_idx = block // value_tiles
                value_lane = tx // lane_group
                k_rank = tx - value_lane * lane_group
                value_idx = value_tile_idx * value_tile + value_lane
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim], "float32")
                q_shared = T.alloc_shared([dim], "float32")
                state_local = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                q_square = T.alloc_var("float32", init=0.0)
                k_square = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim, threads)):
                    kk = tx + i * threads
                    if kk < dim:
                        k_shared[kk] = k[0, 0, head_idx, kk]
                        q_shared[kk] = q[0, 0, head_idx, kk]
                T.sync_threads()

                # L2-normalize q and k; every warp folds the whole row and the
                # butterfly leaves both norms in every thread.
                lane = tx - tx // WARP_LANES * WARP_LANES
                for i in T.serial(T.ceildiv(dim, WARP_LANES)):
                    kk = lane + i * WARP_LANES
                    if kk < dim:
                        q_square += q_shared[kk] * q_shared[kk]
                        k_square += k_shared[kk] * k_shared[kk]
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    q_square += T.shfl_xor(q_square, 1 << stage, width=WARP_LANES)
                    k_square += T.shfl_xor(k_square, 1 << stage, width=WARP_LANES)
                q_gain = T.rsqrt(q_square + l2_eps) * scale
                k_gain = T.rsqrt(k_square + l2_eps)
                for i in T.serial(T.ceildiv(dim, threads)):
                    kk = tx + i * threads
                    if kk < dim:
                        q_shared[kk] *= q_gain
                        k_shared[kk] *= k_gain
                T.sync_threads()

                beta_value = T.sigmoid(beta[0, 0, head_idx])
                decay = T.exp2(g[0, 0, head_idx] * LOG2E)

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    decayed = state[0, head_idx, kk, value_idx] * decay
                    state_local[ii] = decayed
                    old_partial += decayed * k_shared[kk]
                for stage in T.unroll(lane_group_stages):
                    old_partial += T.shfl_xor(old_partial, 1 << stage, width=lane_group)

                value_new = beta_value * (v[0, 0, head_idx, value_idx] - old_partial)
                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    updated = state_local[ii] + k_shared[kk] * value_new
                    state[0, head_idx, kk, value_idx] = updated
                    out_partial += updated * q_shared[kk]
                for stage in T.unroll(lane_group_stages):
                    out_partial += T.shfl_down(out_partial, 1 << stage, width=lane_group)
                if k_rank == 0:
                    o[0, 0, head_idx, value_idx] = out_partial

        return kernel

    return _decode()
