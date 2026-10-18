"""Quantized GEMV tests: decode dots straight from quant bytes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from breezetoken.kernels.cpu.quant_gemv import dequant_gemv, quant_gemv_into
from breezetoken.model.moe import ExpertPool, RouterOutput, route
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused quant GEMV is CUDA-only")
def test_fused_quant_gemv_matches_dequant_reference() -> None:
    """The fused iq2_s/iq4_xs kernels must reproduce the torch dequant GEMV."""
    from breezetoken.kernels.cuda.quant_gemv import (
        iq2s_gemv,
        iq2s_table,
        iq4nl_table,
        iq4xs_gemv,
    )
    from breezetoken.quant.dequant import dequantize

    dev = "cuda"
    # iq2_s, shared x (gate/up layout)
    fmt = FORMATS[GGMLType.IQ2_S]
    rng = np.random.default_rng(11)
    columns, rows = 2048, 24
    raw = torch.from_numpy(
        np.frombuffer(b"".join(MAKERS["iq2_s"](rng) for _ in range(rows * 8)), dtype=np.uint8)
    ).to(dev)
    w = dequantize(raw, fmt, (rows, columns))
    x = torch.randn(columns, device=dev)
    got = iq2s_gemv(x, raw, iq2s_table(dev), rows, columns)
    torch.testing.assert_close(got, w @ x, rtol=1e-5, atol=1e-3)
    # iq4_xs with per-expert x (down layout)
    fmt = FORMATS[GGMLType.IQ4_XS]
    rng = np.random.default_rng(12)
    n, per, columns = 3, 2048, 512
    rows = n * per
    raw = torch.from_numpy(
        np.frombuffer(b"".join(MAKERS["iq4_xs"](rng) for _ in range(rows * 2)), dtype=np.uint8)
    ).to(dev)
    w = dequantize(raw, fmt, (rows, columns))
    xs = torch.randn(n, columns, device=dev)
    got = iq4xs_gemv(xs.reshape(-1), raw, iq4nl_table(dev), rows, columns, rows_per_expert=per)
    want = (w.view(n, per, columns) @ xs.unsqueeze(-1)).squeeze(-1).reshape(-1)
    # random-quant bytes give dots with large magnitudes; reduction-order
    # differences between the kernel and cuBLAS are ~1e-4 relative.
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused expert path is CUDA-only")
def test_expert_pool_fused_matches_torch_path() -> None:
    """ExpertPool CUDA fused path equals the torch fallback on checkpoint bytes."""
    from pathlib import Path

    if not Path(GGUF_PATH).is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    from breezetoken.gguf import GGUFFile

    with GGUFFile(GGUF_PATH) as gguf:
        pool = ExpertPool(gguf, 1, "cuda")
        x = torch.randn(2048, device="cuda")
        torch.manual_seed(9)
        routing = route(x, torch.randn(256, 2048, device="cuda") * 0.02, 8)
        got = pool.forward(0, gguf, x, routing)
        # force the torch path by spoofing the device check
        pool.device = "cpu-fallback"
        want = pool.forward(
            0, gguf, x.cpu(), RouterOutput(routing.weights.cpu(), routing.expert_ids.cpu())
        )
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-3)
