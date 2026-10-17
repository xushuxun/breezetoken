from __future__ import annotations

import numpy as np
import pytest
import torch

from breezetoken.gguf import GGML_F32, GGML_Q8_0, GGUFFile


def test_metadata_and_tensor_shapes(gguf: GGUFFile) -> None:
    assert gguf.arch == "qwen35"
    assert gguf.tensor_info("token_embd.weight")[0] == (248320, 1024)
    assert gguf.tensor_info("blk.0.ssm_a")[0] == (16,)
    assert gguf.tensor_info("blk.0.attn_qkv.weight")[0] == (6144, 1024)


def test_q8_0_dequant_shape_and_dtype(gguf: GGUFFile) -> None:
    tensor = gguf.get_tensor("blk.0.attn_qkv.weight", dtype=torch.float32)
    assert tensor.shape == (6144, 1024)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert tensor.abs().max().item() > 0


def test_f32_tensor(gguf: GGUFFile) -> None:
    tensor = gguf.get_tensor("blk.0.ssm_a", dtype=torch.float32)
    assert tensor.shape == (16,)
    assert tensor.dtype == torch.float32
    assert (tensor <= 0).all()  # ssm_a stores -exp(A_log)


def test_unsupported_type_message(gguf: GGUFFile) -> None:
    # Every tensor in this checkpoint is f32/Q8_0; the path is still exercised
    # through a fake tensor table entry.
    with pytest.raises(NotImplementedError):
        gguf._tensors["fake"] = ((1, 1), 999, 0)
        try:
            gguf.get_tensor("fake")
        finally:
            del gguf._tensors["fake"]


def test_close_invalidates_tensor_access(model_path) -> None:
    file = GGUFFile(model_path)
    file.close()
    with pytest.raises(ValueError, match="closed"):
        file.get_tensor("token_embd.weight", dtype=torch.float32)


def test_q8_0_block_decoder_matches_reference(model_path) -> None:
    """A single Q8_0 block decodes to ``q * scale`` (f16 scale)."""
    block = np.dtype([("scale", "<f2"), ("q", "<i1", (32,))])
    with GGUFFile(model_path) as file:
        shape, ttype, offset = file._tensors["token_embd.weight"]
        assert ttype == GGML_Q8_0
        raw = np.frombuffer(file._require_buf(), dtype=block, count=1, offset=offset)
        expected = raw["q"][0].astype(np.float32) * float(raw["scale"][0])
        tensor = file.get_tensor("token_embd.weight", dtype=torch.float32)
        assert np.allclose(tensor[0, :32].numpy(), expected, atol=1e-4)
    assert GGML_F32 == 0
