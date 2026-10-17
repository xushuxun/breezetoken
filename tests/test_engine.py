from __future__ import annotations

from breezetoken.engine import Engine


def test_engine_delegates_to_model_and_tokenizer(torch_model, tokenizer) -> None:
    engine = Engine(model=torch_model, tokenizer=tokenizer, model_id="test-model")
    assert engine.max_context == torch_model.cfg.max_context
    assert engine.eos_ids == tokenizer.eos_ids

    messages = [{"role": "user", "content": "你好"}]
    assert engine.chat_ids(messages) == tokenizer.chat_ids(messages)
    ids = tokenizer.encode("The capital of France is")
    first = next(iter(engine.generate_stream(ids, 1)))
    assert first == 11751
    assert engine.decode([first]) == " Paris"
    assert engine.token_bytes(first) == b" Paris"
    assert "test-model" in repr(engine)
