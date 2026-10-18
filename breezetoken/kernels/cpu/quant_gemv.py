"""Quantized GEMV: decode dot products straight from GGUF quant bytes.

The reference contract (from docs/qwen36_35b_plan.md phase 2):

    out[i] = dot(dequantize(rows[i]), x)

with no f32 copy of the *whole* weight tensor -- the only temporary is the
requested row tile, dequantized by ``breezetoken.quant.dequant`` and multiplied
with a single torch matvec/matmul.  The same code runs on CPU and CUDA: the
caller stages the raw quant bytes on ``x.device`` and the dequant follows.

This is the phase-2 fallback path (vectorized dequant fused into the GEMV);
a tilelang-C kernel can replace the internals later without changing the
contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from breezetoken.gguf import GGUFFile
from breezetoken.quant.dequant import dequantize
from breezetoken.quant.formats import QuantFormat
from breezetoken.quant.tensor import QuantizedTensor


def dequant_gemv(x: torch.Tensor, raw: torch.Tensor, fmt: QuantFormat, rows: int) -> torch.Tensor:
    """``raw`` quant bytes of ``rows`` rows already on ``x.device``."""
    columns = raw.numel() // rows // fmt.bytes_per_block * fmt.elements_per_block
    w = dequantize(raw, fmt, (rows, columns))
    return w @ x.reshape(-1).to(w.dtype)


def quant_gemv(
    x: torch.Tensor,
    qt: QuantizedTensor,
    gguf: GGUFFile,
    rows: Sequence[int] | int,
    row_count: int | None = None,
) -> torch.Tensor:
    """Dot products of ``x`` with the requested quantized rows.

    ``rows`` is either a single row index (with ``row_count``) or a sequence
    of arbitrary row indices; the result is ``(n, ...)`` on ``x.device``.
    """
    if isinstance(rows, int):
        tile = qt.load_rows(gguf, rows, 1 if row_count is None else row_count, x.device)
        return tile @ x.reshape(-1).to(tile.dtype)
    if not rows:
        raise ValueError("quant_gemv needs at least one row")
    tile = torch.cat([qt.load_rows(gguf, int(r), 1, x.device) for r in rows], dim=0)
    return tile @ x.reshape(-1).to(tile.dtype)


def quant_gemv_into(
    x: torch.Tensor,
    qt: QuantizedTensor,
    gguf: GGUFFile,
    rows: int,
    row_count: int,
    out: torch.Tensor,
) -> None:
    """Scaffold contract: write ``row_count`` dots starting at row ``rows``."""
    tile = qt.load_rows(gguf, rows, row_count, x.device)
    out[:row_count].copy_(tile @ x.reshape(-1).to(tile.dtype))


__all__ = ["dequant_gemv", "quant_gemv", "quant_gemv_into"]
