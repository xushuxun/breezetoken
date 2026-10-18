"""GGUF quant metadata used by the native IQ/K-quant path.

This module intentionally contains no dequantization.  It is the shared table
the reference reader and the tilelang kernels will be built against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

QK_K = 256
QK8_0 = 32


class GGMLType(IntEnum):
    F32 = 0
    Q8_0 = 8
    Q2_K = 10
    Q4_K = 12
    Q6_K = 14
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    BF16 = 30


@dataclass(frozen=True)
class QuantFormat:
    name: str
    type_id: int
    elements_per_block: int
    bytes_per_block: int

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.bytes_per_block / self.elements_per_block

    def row_bytes(self, columns: int) -> int:
        if columns % self.elements_per_block:
            raise ValueError(
                f"{self.name} needs a row divisible by {self.elements_per_block}, got {columns}"
            )
        return (columns // self.elements_per_block) * self.bytes_per_block

    @property
    def is_quantized(self) -> bool:
        return self.elements_per_block > 1


FORMATS: dict[int, QuantFormat] = {
    GGMLType.F32: QuantFormat("f32", 0, 1, 4),
    GGMLType.Q8_0: QuantFormat("q8_0", 8, QK8_0, 34),
    GGMLType.Q2_K: QuantFormat("q2_K", 10, QK_K, 84),
    GGMLType.Q4_K: QuantFormat("q4_K", 12, QK_K, 144),
    GGMLType.Q6_K: QuantFormat("q6_K", 14, QK_K, 210),
    GGMLType.IQ3_S: QuantFormat("iq3_s", 21, QK_K, 110),
    GGMLType.IQ2_S: QuantFormat("iq2_s", 22, QK_K, 82),
    GGMLType.IQ4_XS: QuantFormat("iq4_xs", 23, QK_K, 136),
    GGMLType.BF16: QuantFormat("bf16", 30, 1, 2),
}


def format_for_type(type_id: int) -> QuantFormat:
    try:
        return FORMATS[type_id]
    except KeyError as exc:
        raise NotImplementedError(f"GGML type {type_id} is not in the native plan") from exc
