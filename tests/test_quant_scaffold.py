"""Metadata/scaffold checks for the planned native 35B path."""

from __future__ import annotations

import pytest

from breezetoken.quant.formats import FORMATS, GGMLType, format_for_type


def test_quant_format_table_matches_ggml_blocks() -> None:
    assert FORMATS[GGMLType.Q8_0].row_bytes(64) == 68
    assert FORMATS[GGMLType.Q2_K].row_bytes(256) == 84
    assert FORMATS[GGMLType.Q4_K].row_bytes(256) == 144
    assert FORMATS[GGMLType.Q6_K].row_bytes(256) == 210
    assert FORMATS[GGMLType.IQ3_S].row_bytes(256) == 110
    assert FORMATS[GGMLType.IQ2_S].row_bytes(256) == 82
    assert FORMATS[GGMLType.IQ4_XS].row_bytes(256) == 136
    assert format_for_type(int(GGMLType.IQ3_S)).name == "iq3_s"


def test_quant_scaffold_is_explicitly_unimplemented() -> None:
    from breezetoken.kernels.cpu.quant_gemv import quant_gemv as cpu_gemv
    from breezetoken.quant.dequant import dequantize_row

    with pytest.raises(NotImplementedError, match="phase"):
        dequantize_row(memoryview(b""), format_for_type(21), 256)
    with pytest.raises(NotImplementedError, match="phase"):
        cpu_gemv(None, memoryview(b""), None, format_for_type(21))


def test_qwen36_config_matches_checkpoint_metadata() -> None:
    from pathlib import Path

    from breezetoken.gguf import GGUFFile
    from breezetoken.model.qwen36_moe import Qwen36Config

    path = Path(
        "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/"
        "snapshots/master/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
    )
    if not path.is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    with GGUFFile(path) as gguf:
        cfg = Qwen36Config.from_gguf(gguf)
    assert cfg.hidden == 2048
    assert cfg.n_layers == 40
    assert cfg.n_heads == 16 and cfg.n_kv == 2
    assert cfg.n_experts == 256 and cfg.top_k == 8
    assert cfg.gdn_key_heads == 16 and cfg.gdn_value_heads == 32


def test_engine_rejects_native_moe_without_llama_fallback() -> None:
    from pathlib import Path

    from breezetoken.engine import Engine

    path = Path(
        "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/"
        "snapshots/master/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
    )
    if not path.is_file():
        pytest.skip("Qwen3.6-35B-A3B IQ3_S checkpoint is not available")
    with pytest.raises(NotImplementedError, match="qwen36_35b_plan"):
        Engine.load(path, max_context=128)
