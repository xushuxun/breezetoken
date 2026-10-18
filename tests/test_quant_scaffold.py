"""Scaffold/metadata checks for the native 35B path (phases 1-2 done)."""

from __future__ import annotations

from pathlib import Path

import pytest

from breezetoken.quant.formats import FORMATS, GGMLType, format_for_type

GGUF_PATH = (
    "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/master/"
    "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
)


def test_quant_format_table_matches_ggml_blocks() -> None:
    assert FORMATS[GGMLType.Q8_0].row_bytes(64) == 68
    assert FORMATS[GGMLType.Q2_K].row_bytes(256) == 84
    assert FORMATS[GGMLType.Q4_K].row_bytes(256) == 144
    assert FORMATS[GGMLType.Q6_K].row_bytes(256) == 210
    assert FORMATS[GGMLType.IQ3_S].row_bytes(256) == 110
    assert FORMATS[GGMLType.IQ2_S].row_bytes(256) == 82
    assert FORMATS[GGMLType.IQ4_XS].row_bytes(256) == 136
    assert format_for_type(int(GGMLType.IQ3_S)).name == "iq3_s"


def test_quant_tables_ported_from_ggml() -> None:
    from breezetoken.quant._tables import IQ2S_GRID, IQ3S_GRID, IQ4NL_VALUES

    assert IQ2S_GRID.shape == (1024, 8)
    assert IQ3S_GRID.shape == (512, 4)
    assert IQ4NL_VALUES.tolist() == [
        -127,
        -104,
        -83,
        -65,
        -49,
        -35,
        -22,
        -10,
        1,
        13,
        25,
        38,
        53,
        69,
        89,
        113,
    ]
    # first rows decode from the little-endian C hex literals
    assert IQ2S_GRID[1].tolist() == [43, 8, 8, 8, 8, 8, 8, 8]
    assert IQ3S_GRID[1].tolist() == [3, 1, 1, 1]


def test_qwen36_config_matches_checkpoint_metadata() -> None:
    from breezetoken.gguf import GGUFFile
    from breezetoken.model.qwen36_moe import Qwen36Config

    path = Path(GGUF_PATH)
    if not path.is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    with GGUFFile(path) as gguf:
        cfg = Qwen36Config.from_gguf(gguf)
    assert cfg.hidden == 2048
    assert cfg.n_layers == 40
    assert cfg.n_heads == 16 and cfg.n_kv == 2
    assert cfg.n_experts == 256 and cfg.top_k == 8
    assert cfg.gdn_key_heads == 16 and cfg.gdn_value_heads == 32
    assert cfg.expert_ffn == 512 and cfg.shared_ffn == 512
    assert len(cfg.attn_layers) == 10 and cfg.attn_layers[-1] == 39
