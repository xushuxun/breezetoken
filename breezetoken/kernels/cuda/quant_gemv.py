"""Fused CUDA quantized GEMV: dequant + dot product in one kernel.

One thread block per output row decodes its quant blocks straight from the
staged GGUF bytes (never materializing an f32 weight tile) and reduces the
dot product with ``x`` in shared memory.  Only the formats the 35B
checkpoint's experts use are implemented:

* iq2_s -- expert gate/up, K=2048 (block: fp16 d, qs[64], qh[8], scales[8]);
* iq4_xs -- expert down, K=512 (block: fp16 d, u16 scales_h, scales_l[4], qs[128]).

Both kernels index ``x`` as ``X[(bx // rows_per_expert) * K + col]`` so one
build serves the shared-x gate/up tile (rows_per_expert = all rows) and the
per-expert down projection.  The torch dequant path stays as the CPU/fallback
reference and must match these kernels exactly (tests enforce that).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np
import torch

from breezetoken.gguf import GGUFFile
from breezetoken.quant.formats import QuantFormat
from breezetoken.quant.tensor import QuantizedTensor

_PINS: dict[tuple[str, int], torch.Tensor] = {}
_TABLES: dict[tuple[str, str], torch.Tensor] = {}


def _pinned(device: str, nbytes: int) -> torch.Tensor:
    key = (device, nbytes)
    buf = _PINS.get(key)
    if buf is None:
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        _PINS[key] = buf
    return buf


def _table(name: str, values: np.ndarray, device: str) -> torch.Tensor:
    key = (name, device)
    tab = _TABLES.get(key)
    if tab is None:
        tab = torch.from_numpy(values.copy()).to(device)
        _TABLES[key] = tab
    return tab


def iq2s_table(device: str) -> torch.Tensor:
    from breezetoken.quant._tables import IQ2S_GRID

    return _table("iq2s", IQ2S_GRID, device)


def iq4nl_table(device: str) -> torch.Tensor:
    from breezetoken.quant._tables import IQ4NL_VALUES

    return _table("iq4nl", IQ4NL_VALUES, device)


# --------------------------------------------------------------------------- #
# kernel builders
# --------------------------------------------------------------------------- #


@cache
def _iq2s_gemv(rows: int, columns: int, rows_per_expert: int, threads: int):
    """out[r] = dot(dequant_iq2s(w[r]), x[r // rows_per_expert])."""
    import tilelang
    import tilelang.language as T

    if columns % 256 or threads != 256:
        raise ValueError("iq2s_gemv expects columns % 256 == 0 and 256 threads")
    row_bytes = columns // 256 * 82
    nblocks = columns // 256

    @T.prim_func
    def main(
        X: T.Tensor(((rows // rows_per_expert) * columns,), "float32"),
        W8: T.Tensor((rows, row_bytes), "uint8"),
        W16: T.Tensor((rows, row_bytes // 2), "int16"),
        TAB: T.Tensor((1024, 8), "int8"),
        OUT: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as bx:
            tx = T.get_thread_binding()
            grp = tx // 32  # 8 chunks of 32 elements per 256-block
            rem = tx % 32
            ch = rem // 8  # chunk of 8 within the group
            j = rem % 8
            nib_shift = 4 * (ch // 2)
            qh_shift = 2 * ch
            red = T.alloc_shared((threads,), "float32")
            partial = T.alloc_var("float32", init=0.0)
            x_base = bx // rows_per_expert * columns
            for t in T.serial(nblocks):
                bo = t * (row_bytes // nblocks)
                d = T.Cast("float32", T.reinterpret("float16", W16[bx, bo // 2]))
                qs = T.Cast("int32", W8[bx, bo + 2 + 4 * grp + ch])
                qh = T.Cast("int32", W8[bx, bo + 66 + grp])
                sc = T.Cast("int32", W8[bx, bo + 74 + grp])
                sb = T.Cast("int32", W8[bx, bo + 2 + 32 + 4 * grp + ch])
                idx = qs | (((qh >> qh_shift) & 3) << 8)
                grid = T.Cast("float32", TAB[idx, j])
                db = d * (0.5 + T.Cast("float32", (sc >> nib_shift) & 15)) * 0.25
                sign = 1.0 - 2.0 * T.Cast("float32", (sb >> j) & 1)
                partial += db * grid * sign * X[x_base + t * 256 + tx]
            red[tx] = partial
            T.sync_threads()
            for s in T.serial(8):  # 128, 64, ..., 1
                width = 128 >> s
                if tx < width:
                    red[tx] += red[tx + width]
                T.sync_threads()
            if tx == 0:
                OUT[bx] = red[0]

    return tilelang.compile(main, out_idx=None)


@cache
def _iq4xs_gemv(rows: int, columns: int, rows_per_expert: int, threads: int):
    """out[r] = dot(dequant_iq4xs(w[r]), x[r // rows_per_expert])."""
    import tilelang
    import tilelang.language as T

    if columns % 256 or threads != 256:
        raise ValueError("iq4xs_gemv expects columns % 256 == 0 and 256 threads")
    row_bytes = columns // 256 * 136
    nblocks = columns // 256

    @T.prim_func
    def main(
        X: T.Tensor(((rows // rows_per_expert) * columns,), "float32"),
        W8: T.Tensor((rows, row_bytes), "uint8"),
        W16: T.Tensor((rows, row_bytes // 2), "int16"),
        TAB: T.Tensor((16,), "int8"),
        OUT: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as bx:
            tx = T.get_thread_binding()
            grp = tx // 32  # 8 groups of 32 elements per 256-block
            r = tx % 32
            red = T.alloc_shared((threads,), "float32")
            partial = T.alloc_var("float32", init=0.0)
            x_base = bx // rows_per_expert * columns
            for t in T.serial(nblocks):
                bo = t * (row_bytes // nblocks)
                d = T.Cast("float32", T.reinterpret("float16", W16[bx, bo // 2]))
                sh = T.Cast("int32", W16[bx, bo // 2 + 1]) & 0xFFFF
                sl = T.Cast("int32", W8[bx, bo + 4 + grp // 2])
                ls = ((sl >> (4 * (grp % 2))) & 15) | (((sh >> (2 * grp)) & 3) << 4)
                dl = d * T.Cast("float32", ls - 32)
                qb = T.Cast("int32", W8[bx, bo + 8 + 16 * grp + r % 16])
                nib = T.if_then_else(r < 16, qb & 15, (qb >> 4) & 15)
                partial += dl * T.Cast("float32", TAB[nib]) * X[x_base + t * 256 + tx]
            red[tx] = partial
            T.sync_threads()
            for s in T.serial(8):
                width = 128 >> s
                if tx < width:
                    red[tx] += red[tx + width]
                T.sync_threads()
            if tx == 0:
                OUT[bx] = red[0]

    return tilelang.compile(main, out_idx=None)


def iq2s_gemv(
    x: torch.Tensor,
    raw8: torch.Tensor,
    table: torch.Tensor,
    rows: int,
    columns: int,
    rows_per_expert: int | None = None,
) -> torch.Tensor:
    """Fused iq2_s GEMV over a staged (rows, row_bytes) quant byte tile."""
    rpe = rows if rows_per_expert is None else rows_per_expert
    kernel = _iq2s_gemv(rows, columns, rpe, 256)
    raw8 = raw8.reshape(rows, -1).contiguous()
    out = torch.empty(rows, dtype=torch.float32, device=x.device)
    kernel(x.reshape(-1).contiguous(), raw8, raw8.view(torch.int16), table, out)
    return out


def iq4xs_gemv(
    x: torch.Tensor,
    raw8: torch.Tensor,
    table: torch.Tensor,
    rows: int,
    columns: int,
    rows_per_expert: int | None = None,
) -> torch.Tensor:
    """Fused iq4_xs GEMV over a staged (rows, row_bytes) quant byte tile."""
    rpe = rows if rows_per_expert is None else rows_per_expert
    kernel = _iq4xs_gemv(rows, columns, rpe, 256)
    raw8 = raw8.reshape(rows, -1).contiguous()
    out = torch.empty(rows, dtype=torch.float32, device=x.device)
    kernel(x.reshape(-1).contiguous(), raw8, raw8.view(torch.int16), table, out)
    return out


# --------------------------------------------------------------------------- #
# batched dequant kernels (no dot product; feed batched expert GEMMs)
# --------------------------------------------------------------------------- #


@cache
def _iq2s_dequant(rows: int, columns: int, threads: int):
    """OUT[r, c] = dequant_iq2s(W[r])[c] as bfloat16; one element per thread."""
    import tilelang
    import tilelang.language as T

    row_bytes = columns // 256 * 82
    total = rows * columns

    @T.prim_func
    def main(
        W8: T.Tensor((total // columns, row_bytes), "uint8"),
        W16: T.Tensor((total // columns, row_bytes // 2), "int16"),
        TAB: T.Tensor((1024, 8), "int8"),
        OUT: T.Tensor((total // columns, columns), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            e = bx * threads + tx
            if e < total:
                row = e // columns
                col = e % columns
                t = col // 256  # 256-element block within the row
                within = col % 256
                grp = within // 32
                rem = within % 32
                ch = rem // 8
                j = rem % 8
                bo = t * (row_bytes // (columns // 256))
                d = T.Cast("float32", T.reinterpret("float16", W16[row, bo // 2]))
                qs = T.Cast("int32", W8[row, bo + 2 + 4 * grp + ch])
                qh = T.Cast("int32", W8[row, bo + 66 + grp])
                sc = T.Cast("int32", W8[row, bo + 74 + grp])
                sb = T.Cast("int32", W8[row, bo + 2 + 32 + 4 * grp + ch])
                idx = qs | (((qh >> (2 * ch)) & 3) << 8)
                grid = T.Cast("float32", TAB[idx, j])
                db = d * (0.5 + T.Cast("float32", (sc >> (4 * (ch // 2))) & 15)) * 0.25
                sign = 1.0 - 2.0 * T.Cast("float32", (sb >> j) & 1)
                OUT[row, col] = T.Cast("bfloat16", db * grid * sign)

    return tilelang.compile(main, out_idx=None)


@cache
def _iq4xs_dequant(rows: int, columns: int, threads: int):
    """OUT[r, c] = dequant_iq4xs(W[r])[c] as bfloat16; one element per thread."""
    import tilelang
    import tilelang.language as T

    row_bytes = columns // 256 * 136
    total = rows * columns

    @T.prim_func
    def main(
        W8: T.Tensor((total // columns, row_bytes), "uint8"),
        W16: T.Tensor((total // columns, row_bytes // 2), "int16"),
        TAB: T.Tensor((16,), "int8"),
        OUT: T.Tensor((total // columns, columns), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            e = bx * threads + tx
            if e < total:
                row = e // columns
                col = e % columns
                t = col // 256
                within = col % 256
                grp = within // 32
                r = within % 32
                bo = t * (row_bytes // (columns // 256))
                d = T.Cast("float32", T.reinterpret("float16", W16[row, bo // 2]))
                sh = T.Cast("int32", W16[row, bo // 2 + 1]) & 0xFFFF
                sl = T.Cast("int32", W8[row, bo + 4 + grp // 2])
                ls = ((sl >> (4 * (grp % 2))) & 15) | (((sh >> (2 * grp)) & 3) << 4)
                dl = d * T.Cast("float32", ls - 32)
                qb = T.Cast("int32", W8[row, bo + 8 + 16 * grp + r % 16])
                nib = T.if_then_else(r < 16, qb & 15, (qb >> 4) & 15)
                OUT[row, col] = T.Cast("bfloat16", dl * T.Cast("float32", TAB[nib]))

    return tilelang.compile(main, out_idx=None)


def dequant_tile(raw8: torch.Tensor, fmt: QuantFormat, rows: int, columns: int) -> torch.Tensor:
    """Batched dequant to bf16 on the GPU (torch fallback on CPU)."""
    if raw8.device.type != "cuda":
        from breezetoken.quant.dequant import dequantize

        return dequantize(raw8, fmt, (rows, columns))
    out = torch.empty((rows, columns), dtype=torch.bfloat16, device=raw8.device)
    w8 = raw8.view(rows, -1)
    w16 = w8.view(torch.int16)
    if fmt.name == "iq2_s":
        _iq2s_dequant(rows, columns, 256)(w8, w16, iq2s_table(str(raw8.device)), out)
    elif fmt.name == "iq4_xs":
        _iq4xs_dequant(rows, columns, 256)(w8, w16, iq4nl_table(str(raw8.device)), out)
    else:
        from breezetoken.quant.dequant import dequantize

        return dequantize(raw8, fmt, (rows, columns))
    return out


# --------------------------------------------------------------------------- #
# scaffold contract: staged quant GEMV with torch fallback
# --------------------------------------------------------------------------- #


def quant_gemv(
    x: torch.Tensor,
    qt: QuantizedTensor,
    gguf: GGUFFile,
    rows: Sequence[int] | int,
    row_count: int | None = None,
) -> torch.Tensor:
    """Dot products of ``x`` with quantized rows, dequantized on ``x.device``."""
    device = x.device
    if isinstance(rows, int):
        n = 1 if row_count is None else row_count
        start, stop = qt.byte_range(rows, n)
        nbytes = stop - start
        if device.type == "cuda":
            pin = _pinned(str(device), nbytes)
            np.copyto(pin.numpy(), gguf.bytes_view(start, nbytes))
            raw = pin.to(device, non_blocking=True)
        else:
            raw = torch.from_numpy(np.ascontiguousarray(gguf.bytes_view(start, nbytes)))
        return _fused_or_torch(x, raw, qt.format, n)
    if not rows:
        raise ValueError("quant_gemv needs at least one row")
    return torch.cat(
        [quant_gemv(x, qt, gguf, int(r)) for r in rows],
        dim=0,
    )


def _fused_or_torch(
    x: torch.Tensor, raw: torch.Tensor, fmt: QuantFormat, rows: int
) -> torch.Tensor:
    from breezetoken.kernels.cpu.quant_gemv import dequant_gemv

    columns = raw.numel() // rows // fmt.bytes_per_block * fmt.elements_per_block
    if x.device.type == "cuda":
        if fmt.name == "iq2_s":
            return iq2s_gemv(x, raw, iq2s_table(str(x.device)), rows, columns)
        if fmt.name == "iq4_xs":
            return iq4xs_gemv(x, raw, iq4nl_table(str(x.device)), rows, columns)
    return dequant_gemv(x, raw, fmt, rows)


__all__ = [
    "iq2s_gemv",
    "iq2s_table",
    "iq4nl_table",
    "iq4xs_gemv",
    "quant_gemv",
]
