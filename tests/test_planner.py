"""Placement heuristics: no device choice needed from the caller."""

from __future__ import annotations

import pytest

from breezetoken.gguf import GGUFFile
from breezetoken.model.qwen35 import Qwen35Config
from breezetoken.planner import Hardware, plan

GIB = 1 << 30


def _hardware(**kwargs) -> Hardware:
    base = dict(
        cpu_count=8,
        cpu_ram=32 * GIB,
        cuda=False,
        gpu_name="",
        gpu_total=0,
        gpu_free=0,
        sm_count=0,
        smem_optin=0,
        compute_cap=(0, 0),
    )
    base.update(kwargs)
    return Hardware(**base)


def _cfg(model_path):
    with GGUFFile(model_path) as gguf:
        return Qwen35Config.from_gguf(gguf, max_context=512)


def test_planner_uses_cpu_without_cuda(model_path) -> None:
    placement = plan(_cfg(model_path), _hardware(cpu_ram=32 * GIB), max_context=512)
    assert placement.gpu_layers == 0
    assert placement.head_on_gpu is False
    assert placement.max_context == 512


def test_planner_all_gpu_when_it_fits(model_path) -> None:
    pytest.importorskip("tilelang")
    hw = _hardware(
        cuda=True,
        gpu_name="fake 8GB",
        gpu_total=8 * GIB,
        gpu_free=8 * GIB,
        sm_count=24,
        smem_optin=100 * 1024,
        compute_cap=(8, 9),
    )
    placement = plan(_cfg(model_path), hw, max_context=512)
    assert placement.gpu_layers == 24
    assert placement.head_on_gpu is True


def test_planner_can_cap_gpu_layers(model_path) -> None:
    pytest.importorskip("tilelang")
    hw = _hardware(
        cuda=True,
        gpu_name="fake 8GB",
        gpu_total=8 * GIB,
        gpu_free=8 * GIB,
        sm_count=24,
        smem_optin=100 * 1024,
        compute_cap=(8, 9),
    )
    placement = plan(_cfg(model_path), hw, max_context=512, max_gpu_layers=1)
    assert placement.gpu_layers == 1
    assert placement.head_on_gpu is True


def test_planner_reports_when_cpu_memory_is_short(model_path) -> None:
    with pytest.raises(RuntimeError, match="CPU memory"):
        plan(_cfg(model_path), _hardware(cpu_ram=1 * GIB), max_context=512)
