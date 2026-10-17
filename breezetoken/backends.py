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
    def rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        """Partial rotate-half RoPE on the leading ``rope_dim`` dimensions."""

    @abstractmethod
    def fused_add_rms_norm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h = a + b`` and ``xn = rms_norm(h, w, eps)``, sharing one read."""

    @abstractmethod
    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """``silu(gate) * up``."""

    @abstractmethod
    def gdn_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        scale: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """One gated-delta-rule step; updates ``state`` in place and returns ``o``.

        ``q``/``k``/``v`` are ``(H, D)``; ``g`` (log decay) and ``beta`` (raw
        logits) are ``(H,)``; ``state`` is the f32 ``(H, D, D)`` recurrent state.
        """

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

    def rope(self, x, cos, sin, pos, rope_dim):
        return _torch_rope(x, cos, sin, pos, rope_dim)

    def fused_add_rms_norm(self, a, b, w, eps):
        return _torch_fused_add_rms_norm(a, b, w, eps)

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError(f"silu_mul shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
        return torch.nn.functional.silu(gate.float()) * up.float()

    def gdn_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        scale: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return _torch_gdn_decode(q, k, v, g, beta, state, scale, eps)

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


def _torch_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    p = int(pos.reshape(-1)[0].item())
    rot = x[..., :rope_dim].float()
    tail = x[..., rope_dim:].float()
    half = rope_dim // 2
    turned = rot * cos[p] + torch.cat((-rot[..., half:], rot[..., :half]), dim=-1) * sin[p]
    return torch.cat((turned, tail), dim=-1).to(x.dtype)


def _torch_fused_add_rms_norm(
    a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape:
        if b.numel() == a.numel():
            b = b.reshape(a.shape)
        else:
            raise ValueError(f"fused_add_rms_norm shape mismatch: {a.shape} vs {b.shape}")
    h = a + b
    return h, _torch_rms_norm(h, w, eps)


@torch.no_grad()
def _torch_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference gated-delta-rule step (state is f32 and updated in place)."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 2:
        raise ValueError("q/k/v must all be (heads, dim)")
    heads, dim = q.shape
    if state.shape != (heads, dim, dim):
        raise ValueError(f"state shape {tuple(state.shape)} != {(heads, dim, dim)}")
    qf = q.float()
    kf = k.float()
    vf = v.float()
    qn = qf * torch.rsqrt(qf.pow(2).sum(dim=-1, keepdim=True) + eps) * scale
    kn = kf * torch.rsqrt(kf.pow(2).sum(dim=-1, keepdim=True) + eps)
    decayed = state * torch.exp(g.float()).view(heads, 1, 1)
    recalled = torch.einsum("hdv,hd->hv", decayed, kn)
    delta = (vf - recalled) * torch.sigmoid(beta.float()).unsqueeze(-1)
    state.copy_(decayed + kn.unsqueeze(-1) * delta.unsqueeze(-2))
    return torch.einsum("hdv,hd->hv", state, qn)


def _gdn_launch(heads: int, dim: int, device: str) -> tuple[int, int]:
    """Block shape for the decode kernel (TileOPs' launch heuristic)."""
    sm_count = torch.cuda.get_device_properties(torch.device(device)).multi_processor_count
    if heads * (dim // 32) >= 6 * sm_count:
        return 32, 1
    return 64, 2


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

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.empty((rows, N), dtype=torch.float32, device=device)
            kernel(x, w, out)
            return out

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

        def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            out = torch.empty((rows, N), dtype=torch.float32, device=device)
            kernel(x, w, out)
            return out

        return run


class _CudaRope:
    device = "cuda"
    dtype = torch.float32

    @classmethod
    def supports(cls, rows: int, dim: int, rope_dim: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and rows > 0 and 0 < rope_dim <= dim and rope_dim % 2 == 0

    @classmethod
    def build(cls, rows: int, dim: int, rope_dim: int, cap: int, device: str):
        from breezetoken.kernels.cuda.rope import rope

        kernel = rope(rows, dim, rope_dim, cap)

        def run(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            pos: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty((rows, dim), dtype=torch.float32, device=device)
            kernel(x.reshape(rows, dim), cos, sin, pos.reshape(1).to(torch.int32), out)
            return out

        return run


class _CudaResidRmsNorm:
    device = "cuda"
    dtype = torch.float32

    @classmethod
    def supports(cls, rows: int, N: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and 0 < N <= 8192 and rows > 0

    @classmethod
    def build(cls, rows: int, N: int, eps: float, device: str):
        from breezetoken.kernels.cuda.norm import resid_rms_norm

        threads = 256 if N >= 256 else (128 if N >= 128 else 32)
        kernel = resid_rms_norm(rows, N, eps, threads)

        def run(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor):
            hout = torch.empty((rows, N), dtype=torch.float32, device=device)
            xn = torch.empty((rows, N), dtype=torch.float32, device=device)
            kernel(a, b, w, hout, xn)
            return hout, xn

        return run


class _CpuResidRmsNorm:
    device = "cpu"
    dtype = torch.float32

    @classmethod
    def supports(cls, rows: int, N: int, dtype: torch.dtype) -> bool:
        return dtype == cls.dtype and N > 0 and rows > 0

    @classmethod
    def build(cls, rows: int, N: int, eps: float, device: str):
        from breezetoken.kernels.cpu.norm import resid_rms_norm

        kernel = resid_rms_norm(rows, N, eps)

        def run(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor):
            hout = torch.empty((rows, N), dtype=torch.float32, device=device)
            xn = torch.empty((rows, N), dtype=torch.float32, device=device)
            kernel(a, b, w, hout, xn)
            return hout, xn

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

        def run(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            out = torch.empty(size, dtype=torch.float32, device=device)
            kernel(gate.reshape(-1), up.reshape(-1), out)
            return out

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

        def run(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            out = torch.empty(size, dtype=torch.float32, device=device)
            kernel(gate.reshape(-1), up.reshape(-1), out)
            return out

        return run


def _build_gdn_decode(heads: int, dim: int, scale: float, threads: int, lane_group: int):
    from breezetoken.kernels.cuda.gdn import gdn_decode

    kernel = gdn_decode(heads, dim, scale, threads, lane_group)

    def run(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        out = kernel(
            q.contiguous().view(1, 1, heads, dim),
            k.contiguous().view(1, 1, heads, dim),
            v.contiguous().view(1, 1, heads, dim),
            g.contiguous().view(1, 1, heads),
            beta.contiguous().view(1, 1, heads),
            state.view(1, heads, dim, dim),
        )
        return out.view(heads, dim)

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
        self._rope = [_CudaRope] if family == "cuda" else []
        self._resid = [_CudaResidRmsNorm] if family == "cuda" else [_CpuResidRmsNorm]
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

    def rope(self, x, cos, sin, pos, rope_dim):
        dim = x.shape[-1]
        rows = x.numel() // dim
        for spec in self._rope:
            if not spec.supports(rows, dim, rope_dim, x.dtype):
                continue
            cap = cos.shape[0]
            key = ("rope", spec.__name__, rows, dim, rope_dim, cap, self.device)
            fn = self._cache.get(
                key,
                lambda spec=spec, rows=rows, dim=dim, rope_dim=rope_dim, cap=cap: spec.build(
                    rows, dim, rope_dim, cap, self.device
                ),
            )
            return fn(x.reshape(rows, dim).contiguous(), cos, sin, pos).reshape(x.shape)
        return _torch_rope(x, cos, sin, pos, rope_dim)

    def fused_add_rms_norm(self, a, b, w, eps):
        if a.shape != b.shape:
            if b.numel() == a.numel():
                b = b.reshape(a.shape)
            else:
                raise ValueError(f"fused_add_rms_norm shape mismatch: {a.shape} vs {b.shape}")
        N = a.shape[-1]
        if w.numel() != N:
            raise ValueError(f"fused_add_rms_norm weight shape {tuple(w.shape)} != {N}")
        rows = a.numel() // N
        for spec in self._resid:
            if not spec.supports(rows, N, a.dtype):
                continue
            key = ("resid_rms_norm", spec.__name__, rows, N, eps, self.device)
            fn = self._cache.get(
                key,
                lambda spec=spec, rows=rows, N=N, eps=eps: spec.build(rows, N, eps, self.device),
            )
            hout, xn = fn(
                a.reshape(rows, N).contiguous(),
                b.reshape(rows, N).contiguous(),
                w.reshape(-1),
            )
            return hout.reshape(a.shape), xn.reshape(a.shape)
        return _torch_fused_add_rms_norm(a, b, w, eps)

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

    def gdn_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        scale: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        heads, dim = q.shape
        servable = (
            self._family == "cuda"
            and q.dtype == torch.float32
            and k.dtype == torch.float32
            and v.dtype == torch.float32
            and state.dtype == torch.float32
            and state.shape == (heads, dim, dim)
            and dim in (64, 128)
            and eps == 1e-6
            and state.is_contiguous()
        )
        if servable:
            threads, lane_group = _gdn_launch(heads, dim, self.device)
            key = ("gdn", heads, dim, scale, threads, lane_group, self.device)
            fn = self._cache.get(
                key,
                lambda heads=heads, dim=dim, scale=scale, threads=threads, lane=lane_group: (
                    _build_gdn_decode(heads, dim, scale, threads, lane)
                ),
            )
            return fn(q, k, v, g, beta, state)
        return _torch_gdn_decode(q, k, v, g, beta, state, scale, eps)

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
