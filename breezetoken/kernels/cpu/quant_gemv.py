"""CPU quantized GEMV (phase 2 scaffold)."""

from __future__ import annotations

import torch

from breezetoken.quant.formats import QuantFormat


def quant_gemv(
    x: torch.Tensor,
    rows: memoryview,
    out: torch.Tensor,
    fmt: QuantFormat,
) -> None:
    """``out[r] = dot(dequant(rows[r]), x)`` without materializing all rows."""
    raise NotImplementedError("CPU quant GEMV is phase 2; see docs/qwen36_35b_plan.md")
