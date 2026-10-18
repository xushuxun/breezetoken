"""Reference dequantization interfaces (phase 1, not implemented)."""

from __future__ import annotations

import torch

from breezetoken.quant.formats import QuantFormat


def dequantize_row(raw: memoryview, fmt: QuantFormat, columns: int) -> torch.Tensor:
    """Return one row as f32.

    Phase 1 implements F32/BF16/Q8_0/Q2_K/Q4_K/Q6_K/IQ2_S/IQ3_S/IQ4_XS using
    the block layouts in ``docs/qwen36_35b_plan.md``.
    """
    raise NotImplementedError(f"dequantize_row({fmt.name}) is phase 1; see docs/qwen36_35b_plan.md")
