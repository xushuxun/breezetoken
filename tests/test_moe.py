"""MoE routing tests (phase 4 contract)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from breezetoken.model.moe import ExpertPool, route

GGUF_PATH = (
    "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/master/"
    "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
)


def test_route_softmax_topk_renormalized() -> None:
    torch.manual_seed(3)
    gate = torch.randn(256, 2048) * 0.05
    x = torch.randn(2048)
    out = route(x, gate, 8)
    assert out.weights.shape == (8,)
    assert out.expert_ids.shape == (8,)
    probs = torch.softmax(gate @ x, dim=-1)
    want_w, want_ids = torch.topk(probs, 8)
    want_w = want_w / want_w.sum()
    assert out.expert_ids.tolist() == want_ids.tolist()
    torch.testing.assert_close(out.weights, want_w, rtol=1e-6, atol=1e-7)
    assert out.weights.sum().item() == pytest.approx(1.0, rel=1e-6)


def test_route_weights_descend() -> None:
    torch.manual_seed(11)
    gate = torch.randn(256, 128)
    x = torch.randn(128)
    out = route(x, gate, 8)
    assert (out.weights[:-1] >= out.weights[1:]).all()


def test_expert_pool_shapes_cpu() -> None:
    if not Path(GGUF_PATH).is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    from breezetoken.gguf import GGUFFile

    with GGUFFile(GGUF_PATH) as gguf:
        pool = ExpertPool(gguf, 1, "cpu")
        x = torch.randn(2048)
        r = route(x, torch.randn(256, 2048) * 0.02, 8)
        out = pool.forward(0, gguf, x, r)
        assert out.shape == (2048,)
        assert out.dtype == torch.float32
