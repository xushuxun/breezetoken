"""Quantized GEMV tests: decode dots straight from quant bytes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from breezetoken.kernels.cpu.quant_gemv import dequant_gemv, quant_gemv_into
from breezetoken.quant.formats import FORMATS, GGMLType
from breezetoken.quant.tensor import QuantizedTensor
from tests.test_quant_dequant import MAKERS

GGUF_PATH = "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/master/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"


@pytest.mark.parametrize("name", list(MAKERS))
def test_dequant_gemv_matches_reference_matmul(name: str) -> None:
    """dequant_gemv must equal dequantize-then-matmul on random blocks."""
    type_id = {
        "q8_0": GGMLType.Q8_0,
        "q2_K": GGMLType.Q2_K,
        "q4_K": GGMLType.Q4_K,
        "q6_K": GGMLType.Q6_K,
        "iq2_s": GGMLType.IQ2_S,
        "iq3_s": GGMLType.IQ3_S,
        "iq4_xs": GGMLType.IQ4_XS,
    }[name]
    fmt = FORMATS[type_id]
    rng = np.random.default_rng(7)
    rows = 5
    raw = torch.from_numpy(
        np.frombuffer(b"".join(MAKERS[name](rng) for _ in range(rows)), dtype=np.uint8)
    )
    from breezetoken.quant.dequant import dequantize

    columns = fmt.elements_per_block * (raw.numel() // rows // fmt.bytes_per_block)
    w = dequantize(raw, fmt, (rows, columns))
    x = torch.randn(columns)
    got = dequant_gemv(x, raw, fmt, rows)
    torch.testing.assert_close(got, w @ x, rtol=1e-5, atol=1e-5)


def test_quant_gemv_into_on_checkpoint_rows() -> None:
    """quant_gemv on real checkpoint rows equals the dequant reference."""
    from pathlib import Path

    if not Path(GGUF_PATH).is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    from breezetoken.gguf import GGUFFile

    with GGUFFile(GGUF_PATH) as gguf:
        qt = QuantizedTensor.from_gguf(gguf, "blk.0.attn_qkv.weight")
        w = qt.load_rows(gguf, 100, 64)  # (64, 2048)
        x = torch.randn(2048)
        out = torch.empty(64)
        quant_gemv_into(x, qt, gguf, 100, 64, out)
        torch.testing.assert_close(out, w @ x, rtol=1e-5, atol=1e-3)
