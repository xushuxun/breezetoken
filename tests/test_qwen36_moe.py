"""End-to-end tests for the native Qwen3.6-35B-A3B hybrid decoder.

These need the IQ3_S checkpoint; they skip when it is absent.  The greedy
reference ids were captured from llama.cpp (same GGUF, --temp 0) and must
match bit-for-bit -- dequant, routing, GDN/attention and placement all show
up in the sampled tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

GGUF_PATH = Path(
    "/home/xun/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/master/"
    "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
)

PROMPT = "The capital of France is"
# llama-server /v1/completions (same GGUF, temperature 0): greedy continuation.
LLAMA_GREEDY_IDS: list[int] = [11751, 13, 198, 760, 6511, 314, 9338, 369]
LLAMA_GREEDY_TEXT = " Paris.\nThe capital of France is"

pytestmark = pytest.mark.skipif(not GGUF_PATH.is_file(), reason="35B IQ3_S checkpoint absent")


@pytest.fixture(scope="module")
def engine():
    from breezetoken.engine import Engine

    eng = Engine.load(GGUF_PATH, max_context=256)
    try:
        yield eng
    finally:
        eng.close()


def test_forward_logits_shape(engine) -> None:
    ids = engine.encode(PROMPT)
    logits = engine.model.forward_token(ids[0], 0, logits=True)
    assert logits.shape == (engine.model.cfg.vocab,)
    assert logits.dtype == torch.float32


def test_greedy_matches_llama_cpp(engine) -> None:
    if not LLAMA_GREEDY_IDS:
        pytest.skip("llama.cpp reference ids not captured yet")
    ids = engine.encode(PROMPT)
    out = list(engine.generate_stream(ids, len(LLAMA_GREEDY_IDS), 0.0))
    assert out == LLAMA_GREEDY_IDS


def test_generate_max_new_and_eos(engine) -> None:
    ids = engine.encode(PROMPT)
    out = list(engine.generate_stream(ids, 3, 0.0))
    assert len(out) == 3
    # EOS immediately stops the stream (11751 is the known greedy first token).
    out2 = list(engine.model.generate_stream(ids, 5, 0.0, eos_ids=[11751]))
    assert out2 == []


def test_cpu_only_forward() -> None:
    from breezetoken.engine import Engine

    eng = Engine.load(GGUF_PATH, device="cpu", max_context=64)
    try:
        ids = eng.encode(PROMPT)
        logits = eng.model.forward_token(ids[0], 0, logits=True)
        assert logits.shape == (eng.model.cfg.vocab,)
        assert torch.isfinite(logits).all()
    finally:
        eng.close()
