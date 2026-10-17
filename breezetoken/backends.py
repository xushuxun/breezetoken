"""Compute backends: a torch reference and tilelang kernel backends.

The model only sees this small interface::

    backend.linear(x, w)       # x (..., K), w (K, N) -> (N,)
    backend.rms_norm(x, w, eps)
    backend.silu_mul(gate, up)

A backend owns the kernel cache and the decision to fall back to torch.  The
``torch`` backend is a genuine reference implementation (it imports no
tilelang); the ``tilelang`` backend selects compiled kernels by shape/dtype
and falls back to torch when no kernel serves a call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Backend(ABC):
    """The operations the model needs from a device/kernel family."""

    name: str
    device: str
    weight_dtype: torch.dtype

    @abstractmethod
    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """One decode matvec: ``x`` (..., K) times ``w`` (K, N)."""

    @abstractmethod
    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """RMSNorm over the last dimension, with a flat learned weight."""

    @abstractmethod
    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """``silu(gate) * up``."""

    def close(self) -> None:
        """Release cached kernels; backends must remain usable until called."""
        return None


class TorchBackend(Backend):
    """Pure torch fallback, also the numerical reference."""

    def __init__(self, device: str, weight_dtype: torch.dtype = torch.float32) -> None:
        self.name = "torch"
        self.device = device
        self.weight_dtype = weight_dtype

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != w.shape[0]:
            raise ValueError(f"linear shape mismatch: x {tuple(x.shape)}, w {tuple(w.shape)}")
        return (x.reshape(-1).float() @ w.float()).reshape(-1)

    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        if x.shape[-1] != w.numel():
            raise ValueError(f"rms_norm shape mismatch: x {tuple(x.shape)}, w {tuple(w.shape)}")
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError(f"silu_mul shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
        return torch.nn.functional.silu(gate.float()) * up.float()

    def __repr__(self) -> str:
        return f"TorchBackend(device={self.device!r}, dtype={self.weight_dtype})"


@torch.no_grad()
def _torch_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.reshape(-1).float() @ w.float()).reshape(-1)


def _torch_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.float()) * up.float()


class _KernelCache:
    def __init__(self) -> None:
        self._items: dict[tuple[Any, ...], Callable[..., torch.Tensor]] = {}

    def get(self, key: tuple[Any, ...], build: Callable[[], Callable[..., torch.Tensor]]):
        fn = self._items.get(key)
        if fn is None:
            fn = build()
            self._items[key] = fn
        return fn

    def clear(self) -> None:
        self._items.clear()


# --------------------------------------------------------------------------- #
# Kernel selectors.  Each class is a shape/dtype predicate plus a builder.
# Builders import the tilelang modules lazily so importing breezetoken does not
# pay the tilelang import cost (or require it for the torch backend).
# --------------------------------------------------------------------------- #


class _CudaGemv:
    device = "cuda"
    dtype = torch.bfloat16
    #: (BN, BK, SK, threads) per K; sized for a 99 KB shared-memory ceiling.
    TILES = {
        1024: (256, 64, 8, 256),
        2048: (128, 128, 8, 128),
        3584: (128, 64, 8, 128),
    }

    @classmethod
    def supports(cls, K: int, N: int, dtype: torch.dtype) -> bool:
        if dtype != cls.dtype or K not in cls.TILES:
            return False
        BN, BK, SK, _ = cls.TILES[K]
        return K % (SK * BK) == 0 and N >= BN and N % BN == 0

    @classmethod
    def build(cls, K: int, N: int, device: str):
        from breezetoken.kernels.cuda.gemv import gemv

        BN, BK, SK, threads = cls.TILES[K]
        kernel = gemv(K, N, BN, BK, SK, threads)
        out = torch.empty((SK, N), dtype=torch.float32, device=device)

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            kernel(x.reshape(-1).to(torch.bfloat16), w, out)
            return out.sum(dim=0)

        return run


class _CpuGemv:
    device = "cpu"
    dtype = torch.float32
    BK = 64
    BN = 64

    @classmethod
    def supports(cls, K: int, N: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and K % cls.BK == 0 and N >= cls.BN and N % cls.BN == 0

    @classmethod
    def build(cls, K: int, N: int, device: str):
        from breezetoken.kernels.cpu.gemv import gemv

        kernel = gemv(K, N, cls.BK, cls.BN)
        out = torch.empty(N, dtype=torch.float32, device=device)

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            kernel(x.reshape(-1).float(), w, out)
            return out.clone()

        return run


class _CudaRmsNorm:
    device = "cuda"
    dtype = torch.float32

    @classmethod
    def supports(cls, rows: int, N: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and 0 < N <= 8192 and rows > 0

    @classmethod
    def build(cls, rows: int, N: int, eps: float, device: str):
        from breezetoken.kernels.cuda.norm import rms_norm

        threads = 256 if N >= 256 else (128 if N >= 128 else 32)
        kernel = rms_norm(rows, N, eps, threads)
        out = torch.empty((rows, N), dtype=torch.float32, device=device)

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            kernel(x, w, out)
            return out.clone()

        return run


class _CpuRmsNorm:
    device = "cpu"
    dtype = torch.float32

    @classmethod
    def supports(cls, rows: int, N: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and N > 0 and rows > 0

    @classmethod
    def build(cls, rows: int, N: int, eps: float, device: str):
        from breezetoken.kernels.cpu.norm import rms_norm

        kernel = rms_norm(rows, N, eps)
        out = torch.empty((rows, N), dtype=torch.float32, device=device)

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            kernel(x, w, out)
            return out.clone()

        return run


class _CudaSiluMul:
    device = "cuda"
    dtype = torch.float32

    @classmethod
    def supports(cls, size: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and size > 0

    @classmethod
    def build(cls, size: int, device: str):
        from breezetoken.kernels.cuda.norm import silu_mul

        kernel = silu_mul(size)
        out = torch.empty(size, dtype=torch.float32, device=device)

        def run(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            kernel(gate.reshape(-1), up.reshape(-1), out)
            return out.clone()

        return run


class _CpuSiluMul:
    device = "cpu"
    dtype = torch.float32

    @classmethod
    def supports(cls, size: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and size > 0

    @classmethod
    def build(cls, size: int, device: str):
        from breezetoken.kernels.cpu.norm import silu_mul

        kernel = silu_mul(size)
        out = torch.empty(size, dtype=torch.float32, device=device)

        def run(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            kernel(gate.reshape(-1), up.reshape(-1), out)
            return out.clone()

        return run


class TilelangBackend(Backend):
    """tilelang kernels with a torch fallback for unserved shapes."""

    def __init__(self, device: str, weight_dtype: torch.dtype) -> None:
        self.name = "tilelang"
        self.device = device
        self.weight_dtype = weight_dtype
        family = "cuda" if device.startswith("cuda") else "cpu"
        self._family = family
        self._cache = _KernelCache()
        self._gemv = [_CudaGemv] if family == "cuda" else [_CpuGemv]
        self._rms = [_CudaRmsNorm] if family == "cuda" else [_CpuRmsNorm]
        self._silu = [_CudaSiluMul] if family == "cuda" else [_CpuSiluMul]

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        K, N = w.shape
        if x.shape[-1] != K:
            raise ValueError(f"linear shape mismatch: x {tuple(x.shape)}, w {tuple(w.shape)}")
        for spec in self._gemv:
            if not spec.supports(K, N, w.dtype):
                continue
            key = ("gemv", spec.__name__, K, N, self.device)
            fn = self._cache.get(key, lambda spec=spec, K=K, N=N: spec.build(K, N, self.device))
            return fn(x, w)
        return _torch_linear(x, w)

    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        N = x.shape[-1]
        if w.numel() != N:
            raise ValueError(f"rms_norm shape mismatch: x {tuple(x.shape)}, w {tuple(w.shape)}")
        rows = x.numel() // N
        for spec in self._rms:
            if not spec.supports(rows, N, x.dtype):
                continue
            key = ("rms_norm", spec.__name__, rows, N, eps, self.device)
            fn = self._cache.get(
                key,
                lambda spec=spec, rows=rows, N=N, eps=eps: spec.build(rows, N, eps, self.device),
            )
            out = fn(x.reshape(rows, N).contiguous(), w.reshape(-1))
            return out.reshape(x.shape)
        return _torch_rms_norm(x, w, eps)

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError(f"silu_mul shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
        size = gate.numel()
        for spec in self._silu:
            if not spec.supports(size, gate.dtype):
                continue
            key = ("silu_mul", spec.__name__, size, self.device)
            fn = self._cache.get(key, lambda spec=spec, size=size: spec.build(size, self.device))
            return fn(gate, up)
        return _torch_silu_mul(gate, up)

    def close(self) -> None:
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"TilelangBackend(device={self.device!r}, dtype={self.weight_dtype}, "
            f"family={self._family})"
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        parsed = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"invalid device {device!r}") from exc
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if parsed.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device type {parsed.type!r}")
    return str(parsed)


def _torch_dtype(dtype: str | torch.dtype | None, device: str) -> torch.dtype:
    if dtype is None or dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return aliases[dtype.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"unknown dtype {dtype!r}") from exc


def _tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken install should fall back
        logger.warning("tilelang is unavailable (%s); falling back to torch", exc)
        return False
    return True


def make_backend(
    kind: str = "auto",
    device: str = "auto",
    dtype: str | torch.dtype | None = None,
) -> Backend:
    """Build a backend from CLI-friendly strings."""
    resolved_device = resolve_device(device)
    kind = kind.lower()
    if kind not in ("auto", "torch", "tilelang"):
        raise ValueError(f"unknown backend {kind!r}")

    if kind == "auto":
        # CUDA: the tilelang kernels are the point of the project.  CPU: the
        # single-threaded C backend is slower than torch's vectorised matmul,
        # so auto prefers the reference path unless tilelang is requested.
        if resolved_device.startswith("cuda"):
            kind = "tilelang" if _tilelang_available() else "torch"
        else:
            kind = "torch"

    if kind == "torch":
        # The torch backend is the numerical reference: default to f32 even on
        # CUDA unless the caller explicitly asks for a lower-precision twin.
        weight_dtype = (
            torch.float32 if dtype in (None, "auto") else _torch_dtype(dtype, resolved_device)
        )
        return TorchBackend(resolved_device, weight_dtype)

    if not _tilelang_available():
        raise RuntimeError(
            "backend='tilelang' requested but tilelang could not be imported; "
            "install tilelang or use backend='torch'"
        )

    if resolved_device == "cpu":
        if dtype not in (None, "auto") and _torch_dtype(dtype, resolved_device) != torch.float32:
            raise ValueError("the tilelang CPU backend only supports float32")
        weight_dtype = torch.float32
    else:
        weight_dtype = _torch_dtype(dtype, resolved_device)
        if weight_dtype != torch.bfloat16:
            raise ValueError("the tilelang CUDA backend only supports bfloat16")
    logger.info("using %s backend on %s (weights %s)", kind, resolved_device, weight_dtype)
    return TilelangBackend(resolved_device, weight_dtype)
