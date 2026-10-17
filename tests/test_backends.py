from __future__ import annotations

import pytest
import torch

from breezetoken.backends import make_backend, resolve_device


def test_torch_backend_matches_manual_math() -> None:
    backend = make_backend("torch", "cpu")
    torch.manual_seed(0)
    x = torch.randn(8)
    w = torch.randn(8, 5)
    assert torch.allclose(backend.linear(x, w), x @ w, atol=1e-6)

    x3 = torch.randn(2, 3, 4)
    gamma = torch.randn(4)
    normalized = backend.rms_norm(x3, gamma, 1e-6)
    ref = x3 * torch.rsqrt(x3.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    assert torch.allclose(normalized, ref, atol=1e-6)

    gate = torch.randn(6)
    up = torch.randn(6)
    assert torch.allclose(backend.silu_mul(gate, up), torch.nn.functional.silu(gate) * up)


def test_make_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_device("not-a-device")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
    with pytest.raises(ValueError):
        make_backend("nope", "cpu")


def test_tilelang_cpu_kernels_match_torch() -> None:
    pytest.importorskip("tilelang")
    backend = make_backend("tilelang", "cpu")
    torch.manual_seed(1)
    x = torch.randn(64)
    w = torch.randn(64, 64)
    assert torch.allclose(backend.linear(x, w), x @ w, atol=1e-5)

    x3 = torch.randn(2, 3, 8)
    gamma = torch.randn(8)
    got = backend.rms_norm(x3, gamma, 1e-6)
    ref = x3 * torch.rsqrt(x3.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    assert torch.allclose(got, ref, atol=1e-5)

    gate = torch.randn(32)
    up = torch.randn(32)
    assert torch.allclose(
        backend.silu_mul(gate, up), torch.nn.functional.silu(gate) * up, atol=1e-5
    )
    backend.close()


def test_tilelang_falls_back_for_unserved_shape() -> None:
    pytest.importorskip("tilelang")
    backend = make_backend("tilelang", "cpu")
    x = torch.randn(16)
    w = torch.randn(16, 16)  # N < 64: no CPU GEMV kernel
    assert torch.allclose(backend.linear(x, w), x @ w, atol=1e-6)
    backend.close()


def test_tilelang_cpu_rejects_non_f32() -> None:
    pytest.importorskip("tilelang")
    with pytest.raises(ValueError, match="float32"):
        make_backend("tilelang", "cpu", dtype="bfloat16")


def test_auto_backend_is_usable() -> None:
    backend = make_backend("auto", "cpu")
    assert backend.name in ("torch", "tilelang")
    out = backend.linear(torch.ones(4), torch.ones(4, 4))
    assert out.shape == (4,)
    backend.close()
