"""CUDA-only checks.

The fast kernel test runs automatically when a GPU is visible.  The full model
test is opt-in because it JIT-compiles every kernel shape (about 20s):

    BREEZETOKEN_TEST_CUDA=1 uv run pytest tests/test_cuda.py
"""

from __future__ import annotations

import os

import pytest
import torch

from breezetoken.model.qwen35 import Qwen35

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@cuda
def test_cuda_kernels_match_torch() -> None:
    tilelang = pytest.importorskip("tilelang")
    assert tilelang is not None
    from breezetoken.backends import make_backend

    backend = make_backend("tilelang", "cuda")
    torch.manual_seed(0)
    x = torch.randn(1024, device="cuda")
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    got = backend.linear(x, w)
    ref = x.float() @ w.float()
    assert (got - ref).abs().max().item() < 5e-3 * ref.abs().max().item()

    x3 = torch.randn(2, 3, 256, device="cuda")
    gamma = torch.randn(256, device="cuda")
    got = backend.rms_norm(x3, gamma, 1e-6)
    ref = x3 * torch.rsqrt(x3.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    assert torch.allclose(got, ref, atol=1e-5)

    gate = torch.randn(3584, device="cuda")
    up = torch.randn(3584, device="cuda")
    got = backend.silu_mul(gate, up)
    assert torch.allclose(got, torch.nn.functional.silu(gate) * up, atol=1e-5)

    other = torch.randn_like(x3)
    hout, xn = backend.fused_add_rms_norm(x3, other, gamma, 1e-6)
    ref_h = x3 + other
    ref_xn = ref_h * torch.rsqrt(ref_h.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    assert torch.allclose(hout, ref_h, atol=1e-5)
    assert torch.allclose(xn, ref_xn, atol=1e-5)
    backend.close()


@cuda
def test_cuda_gdn_decode_matches_torch() -> None:
    pytest.importorskip("tilelang")
    from breezetoken.backends import make_backend

    torch.manual_seed(3)
    heads, dim = 16, 128
    q = torch.randn(heads, dim, device="cuda")
    k = torch.randn(heads, dim, device="cuda")
    v = torch.randn(heads, dim, device="cuda")
    g = torch.randn(heads, device="cuda") * 0.1
    beta = torch.randn(heads, device="cuda")
    state_ref = torch.randn(heads, dim, dim, device="cuda")
    state = state_ref.clone()
    scale = dim**-0.5
    ref_backend = make_backend("torch", "cuda")
    kernel_backend = make_backend("tilelang", "cuda")
    ref = ref_backend.gdn_decode(q, k, v, g, beta, state_ref, scale)
    got = kernel_backend.gdn_decode(q, k, v, g, beta, state, scale)
    assert torch.allclose(got, ref, atol=1e-5)
    assert torch.allclose(state, state_ref, atol=1e-5)
    ref_backend.close()
    kernel_backend.close()


@cuda
@pytest.mark.skipif(
    os.environ.get("BREEZETOKEN_TEST_CUDA") != "1",
    reason="set BREEZETOKEN_TEST_CUDA=1 to JIT-compile and run the CUDA model",
)
def test_cuda_model_matches_golden(model_path, tokenizer) -> None:
    pytest.importorskip("tilelang")
    from breezetoken.gguf import GGUFFile

    with GGUFFile(model_path) as gguf:
        model = Qwen35(gguf, backend="tilelang", device="cuda", max_context=1024)
    try:
        ids = tokenizer.encode("The capital of France is")
        assert model.generate(ids, 4, 0.0, eos_ids=tokenizer.eos_ids) == [
            11751,
            13,
            198,
            760,
        ]
    finally:
        model.close()


@cuda
def test_cuda_rope_matches_torch() -> None:
    pytest.importorskip("tilelang")
    from breezetoken.backends import make_backend

    torch.manual_seed(4)
    rows, dim, rope_dim, cap = 8, 256, 64, 128
    x = torch.randn(rows, dim, device="cuda")
    cos = torch.randn(cap, rope_dim, device="cuda")
    sin = torch.randn(cap, rope_dim, device="cuda")
    pos = torch.tensor([17], dtype=torch.int32, device="cuda")
    ref_backend = make_backend("torch", "cuda")
    kernel_backend = make_backend("tilelang", "cuda")
    ref = ref_backend.rope(x, cos, sin, pos, rope_dim)
    got = kernel_backend.rope(x, cos, sin, pos, rope_dim)
    assert torch.allclose(got, ref, atol=1e-5)
    ref_backend.close()
    kernel_backend.close()
