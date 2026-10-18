"""CUDA MoE routing/expert kernels (phase 4 scaffold)."""

from __future__ import annotations


def router_topk(*args, **kwargs):
    raise NotImplementedError("MoE routing is phase 4; see docs/qwen36_35b_plan.md")


def expert_gemv(*args, **kwargs):
    raise NotImplementedError("MoE expert GEMV is phase 4; see docs/qwen36_35b_plan.md")
