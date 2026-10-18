"""MoE data contracts for the native Qwen3.6 path (scaffold).

The router produces top-8 expert ids and weights.  The expert tensor layout is
``(experts, out, in)``, so no transpose is needed for gate/up/down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouterOutput:
    weights: torch.Tensor  # (top_k,)
    expert_ids: torch.Tensor  # (top_k,)


@dataclass(frozen=True)
class ExpertBatch:
    ids: torch.Tensor
    weights: torch.Tensor


def route(x: torch.Tensor, gate_inp: torch.Tensor, top_k: int) -> RouterOutput:
    raise NotImplementedError("native router is phase 4; see docs/qwen36_35b_plan.md")
