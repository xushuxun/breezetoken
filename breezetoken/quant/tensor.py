"""Descriptor for a quantized GGUF tensor without materializing it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from breezetoken.gguf import GGUFFile
from breezetoken.quant.dequant import dequantize
from breezetoken.quant.formats import QuantFormat, format_for_type


@dataclass(frozen=True)
class QuantizedTensor:
    """A row-major tensor whose rows are quant blocks in the mmap."""

    name: str
    shape: tuple[int, ...]
    format: QuantFormat
    offset: int
    row_bytes: int

    @property
    def rows(self) -> int:
        if not self.shape:
            return 1
        return int(1 if len(self.shape) == 1 else math.prod(shape_leading(self.shape)))

    @property
    def columns(self) -> int:
        return self.shape[-1]

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, name: str) -> QuantizedTensor:
        shape, type_id, offset = gguf.tensor_layout(name)
        fmt = format_for_type(type_id)
        return cls(name, shape, fmt, offset, fmt.row_bytes(shape[-1]))

    def byte_range(self, row: int = 0, row_count: int | None = None) -> tuple[int, int]:
        """[start, stop) byte offsets in the GGUF mapping for a row tile."""
        n = self.rows - row if row_count is None else row_count
        if not 0 <= row <= self.rows or n < 0 or row + n > self.rows:
            raise IndexError((row, row_count, self.rows))
        start = self.offset + row * self.row_bytes
        return start, start + n * self.row_bytes

    def bytes_view(self, gguf: GGUFFile, row: int = 0, row_count: int | None = None) -> np.memmap:
        """Zero-copy uint8 view of a row tile (expert, out) rows included."""
        start, stop = self.byte_range(row, row_count)
        return gguf.bytes_view(start, stop - start)

    def load_rows(
        self,
        gguf: GGUFFile,
        row: int = 0,
        row_count: int | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Dequantize a row tile to an f32 tensor of (n, columns) on ``device``."""
        view = self.bytes_view(gguf, row, row_count)
        raw = torch.from_numpy(np.ascontiguousarray(view))
        return dequantize(raw, self.format, (view.nbytes // self.row_bytes, self.columns)).to(
            device
        )

    def row_slice(self, row: int) -> tuple[int, int]:
        """Byte range of one row, relative to the mapping start (scaffold API)."""
        if not 0 <= row < self.rows:
            raise IndexError(row)
        start = self.offset + row * self.row_bytes
        return start, start + self.row_bytes


def shape_leading(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[:-1]


__all__ = ["QuantizedTensor"]
