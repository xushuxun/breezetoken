"""Descriptor for a quantized GGUF tensor without materializing it."""

from __future__ import annotations

import math
from dataclasses import dataclass

from breezetoken.gguf import GGUFFile
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
        return int(1 if len(self.shape) == 1 else math.prod(self.shape[:-1]))

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, name: str) -> QuantizedTensor:
        layout = getattr(gguf, "tensor_layout", None)
        if layout is None:
            raise NotImplementedError("GGUFFile.tensor_layout is part of phase 1")
        shape, type_id, offset = layout(name)
        fmt = format_for_type(type_id)
        return cls(name, shape, fmt, offset, fmt.row_bytes(shape[-1]))

    def row_slice(self, row: int) -> tuple[int, int]:
        if not 0 <= row < self.rows:
            raise IndexError(row)
        start = self.offset + row * self.row_bytes
        return start, start + self.row_bytes
