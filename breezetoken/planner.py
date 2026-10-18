"""Hardware/model-aware placement and default parameters.

The engine no longer makes the caller choose CPU or GPU.  It profiles the
machine, estimates the bytes each decoder layer needs (weights + KV/GDN state),
and places the longest possible GPU prefix.  What does not fit runs on the CPU
through the torch backend; the residual stream crosses devices once at the
boundary.

The default for a model that fits entirely in VRAM is therefore all-GPU -- the
fastest configuration -- while a larger model degrades to hybrid instead of
failing.  CPU-only machines still work with the same API.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MIB = 1 << 20
_GIB = 1 << 30

#: Preferred context for the auto path; lowered when memory says otherwise.
_AUTO_CONTEXT = 8192


def _available_ram() -> int:
    """Linux MemAvailable (file cache is reclaimable); fall back to sysconf."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page * pages)
    except (ValueError, OSError, AttributeError):
        return 0


@dataclass(frozen=True)
class Hardware:
    cpu_count: int
    cpu_ram: int
    cuda: bool
    gpu_name: str = ""
    gpu_total: int = 0
    gpu_free: int = 0
    sm_count: int = 0
    smem_optin: int = 0
    compute_cap: tuple[int, int] = (0, 0)

    @classmethod
    def detect(cls) -> Hardware:
        cuda = torch.cuda.is_available()
        gpu_name = ""
        total = free = sm = smem = 0
        cap = (0, 0)
        if cuda:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            total = int(props.total_memory)
            free = int(torch.cuda.mem_get_info(0)[0])
            sm = int(props.multi_processor_count)
            smem = int(getattr(props, "shared_memory_per_block_optin", 0))
            cap = (int(props.major), int(props.minor))
        ram = _available_ram()
        return cls(
            cpu_count=os.cpu_count() or 1,
            cpu_ram=ram,
            cuda=cuda,
            gpu_name=gpu_name,
            gpu_total=total,
            gpu_free=free,
            sm_count=sm,
            smem_optin=smem,
            compute_cap=cap,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Placement:
    max_context: int
    gpu_layers: int
    head_on_gpu: bool
    gpu_bytes: int
    cpu_bytes: int
    reason: str
    gpu_name: str = ""

    def summary(self, total_layers: int) -> str:
        if self.gpu_layers == 0:
            where = "cpu"
        elif self.gpu_layers >= total_layers:
            where = "cuda"
        else:
            where = f"hybrid[{self.gpu_layers}/{total_layers} gpu]"
        head = "cuda" if self.head_on_gpu else "cpu"
        return f"{where} head={head} ctx={self.max_context}"


def _layer_elements(cfg) -> tuple[list[int], list[int], list[int]]:
    """(weight elements, KV bytes/token, fixed recurrent bytes) per layer."""
    hidden, ffn = cfg.hidden, cfg.ffn
    mlp = 3 * hidden * ffn
    common_norm = 2 * hidden
    weights: list[int] = []
    kv_bytes: list[int] = []
    fixed_bytes: list[int] = []
    for i in range(cfg.n_layers):
        if i in cfg.attn_layers:
            q = hidden * cfg.n_heads * 2 * cfg.head_dim
            kv = hidden * cfg.n_kv * cfg.head_dim
            o = cfg.n_heads * cfg.head_dim * hidden
            weights.append(mlp + common_norm + q + 2 * kv + o + 2 * cfg.head_dim)
            kv_bytes.append(2 * cfg.n_kv * cfg.head_dim * 4)
            fixed_bytes.append(0)
        else:
            qkv = hidden * 3 * cfg.ssm_inner
            z = hidden * cfg.ssm_inner
            out = cfg.ssm_inner * hidden
            ab = 2 * hidden * cfg.ssm_heads
            conv = 3 * cfg.ssm_inner * cfg.conv_k
            small = 2 * cfg.ssm_heads + cfg.ssm_state
            weights.append(mlp + common_norm + qkv + z + out + ab + conv + small)
            kv_bytes.append(0)
            fixed_bytes.append(
                cfg.ssm_heads * cfg.ssm_state * cfg.ssm_state * 4
                + 3 * cfg.ssm_inner * (cfg.conv_k - 1) * 4
            )
    return weights, kv_bytes, fixed_bytes


def _tilelang_usable() -> bool:
    try:
        import tilelang  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def plan(
    cfg,
    hw: Hardware | None = None,
    *,
    max_context: int | None = None,
    max_gpu_layers: int | None = None,
) -> Placement:
    """Choose context, GPU layer prefix and head placement.

    ``max_context`` is a hard request (raises when it cannot fit); ``None`` lets
    the planner pick the preferred value and lower it under memory pressure.
    """
    hw = hw or Hardware.detect()
    weights, kv_bytes, fixed_bytes = _layer_elements(cfg)
    if max_context is not None and max_context <= 0:
        raise ValueError("max_context must be positive")
    if max_gpu_layers is not None and max_gpu_layers < 0:
        raise ValueError("max_gpu_layers must be >= 0")
    explicit_context = max_context is not None
    requested = min(int(max_context or _AUTO_CONTEXT), cfg.context_length)

    candidates = (
        [requested]
        if explicit_context
        else [c for c in (_AUTO_CONTEXT, 4096, 2048, 1024) if c <= cfg.context_length]
    )
    if not candidates:
        candidates = [min(cfg.context_length, 1024)]

    gpu_usable = hw.cuda and hw.sm_count > 0 and _tilelang_usable()
    last_error = ""
    for cap in candidates:
        try:
            return _plan_at(
                cfg, hw, weights, kv_bytes, fixed_bytes, cap, gpu_usable, max_gpu_layers
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if explicit_context:
                raise
            logger.info("context %d does not fit: %s", cap, exc)
    raise RuntimeError(last_error or "no placement fits this model")


def _plan_at(
    cfg,
    hw: Hardware,
    weights: list[int],
    kv_bytes: list[int],
    fixed_bytes: list[int],
    cap: int,
    gpu_usable: bool,
    max_gpu_layers: int | None,
) -> Placement:
    f32 = 4
    bf16 = 2
    n = cfg.n_layers
    head_bytes_bf16 = cfg.vocab * cfg.hidden * bf16
    head_bytes_f32 = cfg.vocab * cfg.hidden * f32

    # GPU: 1 GiB is the CUDA context/tilelang workspace reserve.
    reserve = 1 * _GIB
    gpu_budget = max(0, min(hw.gpu_free, int(hw.gpu_total * 0.92)) - reserve)
    if max_gpu_layers is not None:
        gpu_budget = gpu_budget if max_gpu_layers > 0 else 0

    gpu_layers = 0
    head_on_gpu = False
    gpu_bytes = 0
    if gpu_usable:
        head_fits = head_bytes_bf16 <= gpu_budget
        # Prefer a GPU head; if it leaves no room for a layer, let the head go
        # to the CPU so the GPU can still hold at least one decoder layer.
        for candidate_head in (True, False) if head_fits else (False,):
            budget = gpu_budget - (head_bytes_bf16 if candidate_head else 0)
            used = 0
            count = 0
            for i in range(n):
                need = weights[i] * bf16 + kv_bytes[i] * cap + fixed_bytes[i]
                if used + need > budget:
                    break
                used += need
                count += 1
            if max_gpu_layers is not None:
                count = min(count, max_gpu_layers)
                used = sum(
                    weights[i] * bf16 + kv_bytes[i] * cap + fixed_bytes[i] for i in range(count)
                )
            if count > 0 or (candidate_head and max_gpu_layers == 0):
                gpu_layers = count
                head_on_gpu = candidate_head
                gpu_bytes = used + (head_bytes_bf16 if candidate_head else 0)
                break
        if gpu_layers == 0 and not head_on_gpu:
            gpu_usable = False

    cpu_bytes = sum(weights[i] * f32 for i in range(gpu_layers, n))
    cpu_bytes += 0 if head_on_gpu else head_bytes_f32
    cpu_bytes += sum(kv_bytes[i] * cap + fixed_bytes[i] for i in range(gpu_layers, n))
    if hw.cpu_ram and cpu_bytes > int(hw.cpu_ram * 0.85):
        raise RuntimeError(
            f"CPU memory: need {cpu_bytes / _GIB:.1f} GiB, available {hw.cpu_ram / _GIB:.1f} GiB"
        )

    if not gpu_usable:
        where = "cpu"
        reason = "no usable CUDA tilelang backend"
    elif gpu_layers >= n:
        where = "all GPU"
        reason = "model + KV/state fit in VRAM"
    elif gpu_layers == 0:
        where = "cpu"
        reason = "VRAM too small for a decoder layer"
    else:
        where = f"hybrid {gpu_layers}/{n} layers on GPU"
        reason = "VRAM holds a GPU prefix; the rest runs on torch CPU"
    logger.info(
        "placement: %s, head=%s, context=%d, gpu=%.2fGiB cpu=%.2fGiB (%s)",
        where,
        "gpu" if head_on_gpu else "cpu",
        cap,
        gpu_bytes / _GIB,
        cpu_bytes / _GIB,
        reason,
    )
    return Placement(
        max_context=cap,
        gpu_layers=gpu_layers,
        head_on_gpu=head_on_gpu,
        gpu_bytes=gpu_bytes,
        cpu_bytes=cpu_bytes,
        reason=reason,
        gpu_name=hw.gpu_name,
    )
