"""Qwen3.6-35B-A3B native loader/model (scaffold only).

No weights are materialized here.  The plan is in docs/qwen36_35b_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from breezetoken.gguf import GGUFFile


@dataclass(frozen=True)
class Qwen36Config:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    rope_dim: int
    context_length: int
    n_experts: int
    top_k: int
    expert_ffn: int
    shared_ffn: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_k_dim: int
    gdn_v_dim: int
    gdn_conv: int
    nextn_layers: int

    @classmethod
    def from_gguf(cls, gguf: GGUFFile) -> Qwen36Config:
        md = gguf.metadata
        return cls(
            hidden=int(md["qwen35moe.embedding_length"]),
            n_layers=int(md["qwen35moe.block_count"])
            - int(md.get("qwen35moe.nextn_predict_layers", 0)),
            n_heads=int(md["qwen35moe.attention.head_count"]),
            n_kv=int(md["qwen35moe.attention.head_count_kv"]),
            head_dim=int(md["qwen35moe.attention.key_length"]),
            rope_dim=int(md["qwen35moe.rope.dimension_count"]),
            context_length=int(md["qwen35moe.context_length"]),
            n_experts=int(md["qwen35moe.expert_count"]),
            top_k=int(md["qwen35moe.expert_used_count"]),
            expert_ffn=int(md["qwen35moe.expert_feed_forward_length"]),
            shared_ffn=int(md["qwen35moe.expert_shared_feed_forward_length"]),
            gdn_key_heads=int(md["qwen35moe.ssm.group_count"]),
            gdn_value_heads=int(md["qwen35moe.ssm.inner_size"])
            // int(md["qwen35moe.ssm.state_size"]),
            gdn_k_dim=int(md["qwen35moe.ssm.state_size"]),
            gdn_v_dim=int(md["qwen35moe.ssm.state_size"]),
            gdn_conv=int(md["qwen35moe.ssm.conv_kernel"]),
            nextn_layers=int(md.get("qwen35moe.nextn_predict_layers", 0)),
        )


class Qwen36Moe:
    """Native 35B decoder; intentionally unimplemented until phase 3."""

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError(
            "native Qwen3.6-35B-A3B is not implemented; see docs/qwen36_35b_plan.md"
        )
