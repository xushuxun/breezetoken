"""Qwen3.5 text configuration and per-layer decode state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from breezetoken.gguf import GGUFFile

#: Qwen3.5 text config defaults; every value is overridden from GGUF metadata.
_DEFAULT_CONTEXT = 8192


@dataclass(frozen=True)
class Qwen35Config:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    ffn: int
    vocab: int
    eps: float
    l2_eps: float
    ssm_inner: int
    ssm_state: int
    ssm_heads: int
    conv_k: int
    full_attention_interval: int
    context_length: int
    max_context: int
    nextn_layers: int = 0

    @property
    def attn_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if (i + 1) % self.full_attention_interval == 0)

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, max_context: int | None = None) -> Qwen35Config:
        md = gguf.metadata
        n_heads = int(md["qwen35.attention.head_count"])
        head_dim = int(
            md.get("qwen35.attention.key_length", int(md["qwen35.embedding_length"]) // n_heads)
        )
        nextn = int(md.get("qwen35.nextn_predict_layers", 0))
        n_layers = int(md["qwen35.block_count"]) - nextn
        context_length = int(md.get("qwen35.context_length", _DEFAULT_CONTEXT))
        requested = _DEFAULT_CONTEXT if max_context is None else int(max_context)
        if requested <= 0:
            raise ValueError("max_context must be positive")
        context = min(context_length, requested) if context_length > 0 else requested
        cfg = cls(
            hidden=int(md["qwen35.embedding_length"]),
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv=int(md["qwen35.attention.head_count_kv"]),
            head_dim=head_dim,
            rope_dim=int(md.get("qwen35.rope.dimension_count", head_dim)),
            rope_theta=float(md["qwen35.rope.freq_base"]),
            ffn=int(md["qwen35.feed_forward_length"]),
            vocab=int(gguf.tensor_info("token_embd.weight")[0][0]),
            eps=float(md.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6)),
            l2_eps=1e-6,
            ssm_inner=int(md["qwen35.ssm.inner_size"]),
            ssm_state=int(md["qwen35.ssm.state_size"]),
            ssm_heads=int(md["qwen35.ssm.group_count"]),
            conv_k=int(md["qwen35.ssm.conv_kernel"]),
            full_attention_interval=int(md.get("qwen35.full_attention_interval", 4)),
            context_length=context_length,
            max_context=context,
            nextn_layers=nextn,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.full_attention_interval <= 0:
            raise ValueError("full_attention_interval must be positive")
        if not self.attn_layers:
            raise ValueError("model has no full-attention layers")
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError(f"invalid rope_dim {self.rope_dim}")
        if not 0 < self.max_context <= self.context_length:
            raise ValueError(f"max_context {self.max_context} outside (0, {self.context_length}]")
        if self.n_heads % self.n_kv:
            raise ValueError("n_heads must be divisible by n_kv")
        if self.ssm_heads * self.ssm_state != self.ssm_inner:
            raise ValueError("ssm group_count * state_size must equal ssm.inner_size")


class KVCache:
    """One layer's full-attention KV cache on its placement device."""

    def __init__(self, cfg: Qwen35Config, device: str, dtype: torch.dtype):
        shape = (cfg.max_context, cfg.n_kv * cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def reset(self) -> None:
        # Positions are overwritten before they are read, so clearing is not
        # required; keep the method for an explicit lifecycle.
        pass


class GDNState:
    """Per-GDN-layer convolution window and recurrent state."""

    def __init__(self, cfg: Qwen35Config, device: str):
        self.conv = torch.zeros(
            3 * cfg.ssm_inner, cfg.conv_k - 1, device=device, dtype=torch.float32
        )
        self.rec = torch.zeros(
            cfg.ssm_heads, cfg.ssm_state, cfg.ssm_state, device=device, dtype=torch.float32
        )

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()


__all__ = ["GDNState", "KVCache", "Qwen35Config"]
