"""CUDA quantized GEMV: same contract as the CPU path.

The quant bytes are staged to the GPU and dequantized there with the same
torch decoder, so arithmetic matches the CPU reference exactly; only the
placement differs.  Row staging goes through a reusable pinned host buffer
when the rows are contiguous (the MoE per-expert layout).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from breezetoken.gguf import GGUFFile
from breezetoken.kernels.cpu.quant_gemv import dequant_gemv
from breezetoken.quant.tensor import QuantizedTensor

_PINS: dict[tuple[str, int], torch.Tensor] = {}


def _pinned(device: str, nbytes: int) -> torch.Tensor:
    key = (device, nbytes)
    buf = _PINS.get(key)
    if buf is None:
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        _PINS[key] = buf
    return buf


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
        return dequant_gemv(x, raw, qt.format, n)
    if not rows:
        raise ValueError("quant_gemv needs at least one row")
    return torch.cat(
        [quant_gemv(x, qt, gguf, int(r)) for r in rows],
        dim=0,
    )


__all__ = ["quant_gemv"]
