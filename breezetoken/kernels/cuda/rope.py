"""Partial RoPE over a batch of rows (q or k heads).

Only the leading ``rope_dim`` dimensions rotate; the rest pass through.  The
position is a device int32 so one kernel serves every decode step.
"""

from functools import cache

import tilelang
import tilelang.language as T


@cache
def rope(rows: int, dim: int, rope_dim: int, cap: int, threads: int = 128):
    if rope_dim % 2:
        raise ValueError("rope_dim must be even")
    half = rope_dim // 2

    @tilelang.jit(compile_flags=["-O3"])
    def _rope():
        @T.prim_func
        def kernel(
            X: T.Tensor((rows, dim), "float32"),
            Cos: T.Tensor((cap, rope_dim), "float32"),
            Sin: T.Tensor((cap, rope_dim), "float32"),
            Pos: T.Tensor((1,), "int32"),
            Y: T.Tensor((rows, dim), "float32"),
        ):
            with T.Kernel(rows, threads=threads) as row:
                for d in T.Parallel(dim):
                    with T.If(d < rope_dim):
                        with T.Then():
                            partner = T.if_then_else(d < half, d + half, d - half)
                            sign = T.if_then_else(d < half, -1.0, 1.0)
                            Y[row, d] = (
                                X[row, d] * Cos[Pos[0], d] + sign * X[row, partner] * Sin[Pos[0], d]
                            )
                        with T.Else():
                            Y[row, d] = X[row, d]

        return kernel

    return _rope()
