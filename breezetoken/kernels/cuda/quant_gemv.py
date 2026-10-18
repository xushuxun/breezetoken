"""CUDA quantized GEMV (phase 2 scaffold)."""

from __future__ import annotations

import torch

from breezetoken.quant.formats import QuantFormat


def quant_gemv(
    x: torch.Tensor,
    rows: torch.Tensor,
    out: torch.Tensor,
    fmt: QuantFormat,
) -> None:
    """Decode quant blocks in a tilelang kernel and accumulate one output row."""
    raise NotImplementedError("CUDA quant GEMV is phase 2; see docs/qwen36_35b_plan.md")
