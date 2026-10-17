from __future__ import annotations

import os
import threading

import pytest

from breezetoken.model.qwen35 import Qwen35

GOLDEN_PROMPT = "The capital of France is"
GOLDEN_IDS = [11751, 13, 198, 760]


def test_greedy_matches_hf_golden(torch_model: Qwen35, tokenizer) -> None:
    ids = tokenizer.encode(GOLDEN_PROMPT)
    out = torch_model.generate(ids, 4, 0.0, eos_ids=tokenizer.eos_ids)
    assert out == GOLDEN_IDS


def test_prefill_skips_logits_except_last_prompt_token(
    torch_model: Qwen35, tokenizer, monkeypatch
) -> None:
    model = torch_model
    original = model.forward_token
    calls: list[bool] = []

    def spy(token, pos, *, logits=True):
        calls.append(logits)
        return original(token, pos, logits=logits)

    monkeypatch.setattr(model, "forward_token", spy)
    ids = tokenizer.encode(GOLDEN_PROMPT)
    model.generate(ids, 2, 0.0, eos_ids=tokenizer.eos_ids)
    # five prompt tokens: only the last asks for logits; then one forward for
    # the second generated token.
    assert calls == [False, False, False, False, True, True]


def test_empty_prompt_is_rejected(torch_model: Qwen35) -> None:
    with pytest.raises(ValueError, match="at least one token"):
        list(torch_model.generate_stream([], 1))


def test_context_limit_is_rejected(torch_model: Qwen35) -> None:
    model = torch_model
    with pytest.raises(ValueError, match="max_context"):
        list(model.generate_stream([0] * model.cfg.max_context, 1))


def test_cancel_before_generation(torch_model: Qwen35) -> None:
    cancel = threading.Event()
    cancel.set()
    assert list(torch_model.generate_stream([1], 3, cancel=cancel)) == []


@pytest.mark.skipif(
    os.environ.get("BREEZETOKEN_TEST_TILELANG") != "1",
    reason="set BREEZETOKEN_TEST_TILELANG=1 to compile/run the CPU tilelang path",
)
def test_tilelang_cpu_matches_torch(gguf, tokenizer) -> None:
    pytest.importorskip("tilelang")
    model = Qwen35(gguf, backend="tilelang", device="cpu")
    try:
        ids = tokenizer.encode(GOLDEN_PROMPT)
        assert model.generate(ids, 4, 0.0, eos_ids=tokenizer.eos_ids) == GOLDEN_IDS
    finally:
        model.close()


def test_eos_stops_before_yield(torch_model: Qwen35, tokenizer) -> None:
    ids = tokenizer.encode(GOLDEN_PROMPT)
    assert list(torch_model.generate_stream(ids, 4, eos_ids={GOLDEN_IDS[0]})) == []


def test_seeded_sampling_is_reproducible(torch_model: Qwen35, tokenizer) -> None:
    ids = tokenizer.encode(GOLDEN_PROMPT)
    first = torch_model.generate(ids, 2, temperature=1.0, seed=1234)
    second = torch_model.generate(ids, 2, temperature=1.0, seed=1234)
    assert first == second


def test_cancel_mid_generation(torch_model: Qwen35, tokenizer) -> None:
    ids = tokenizer.encode(GOLDEN_PROMPT)
    cancel = threading.Event()
    stream = torch_model.generate_stream(ids, 8, cancel=cancel)
    first = next(stream)
    assert isinstance(first, int)
    cancel.set()
    assert list(stream) == []
