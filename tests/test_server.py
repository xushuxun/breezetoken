from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from breezetoken.server import Settings, create_app


class _Backend:
    name = "torch"


class _Model:
    backend = _Backend()
    device = "cpu"


class FakeEngine:
    """A deterministic engine that avoids loading a real model."""

    def __init__(
        self,
        output: list[bytes] | None = None,
        *,
        max_context: int = 64,
        fail_after: int | None = None,
    ) -> None:
        self.output = output or [b"a", b"b", b"c"]
        self.max_context = max_context
        self.fail_after = fail_after
        self.model_id = "fake-model"
        self.model = _Model()
        self.eos_ids = frozenset()
        self.closed = False

    def chat_ids(self, messages, **kwargs):
        content = messages[-1].get("content")
        if isinstance(content, list) and any(item.get("type") == "image" for item in content):
            raise ValueError("multimodal content is not supported by the qwen35 text engine")
        return [1, 2]

    def encode(self, text: str) -> list[int]:
        return [1, 2] if text else []

    def generate_stream(self, ids, max_new, temperature=0.0, **kwargs):
        cancel = kwargs.get("cancel")
        for index, _ in enumerate(self.output[:max_new]):
            if cancel is not None and cancel.is_set():
                return
            if self.fail_after is not None and index == self.fail_after:
                raise RuntimeError("fake failure")
            yield 100 + index

    def token_bytes(self, token_id: int) -> bytes:
        return self.output[token_id - 100]

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _client(engine: FakeEngine, **settings: Any) -> TestClient:
    app = create_app(Settings(model_path="unused", warmup=False, **settings), engine=engine)
    return TestClient(app)


def _sse_events(body: str) -> list[dict[str, Any]]:
    events = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def test_health_and_models() -> None:
    with _client(FakeEngine()) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "ok" and health["model"] == "fake-model"
        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "fake-model"
        assert client.get("/").status_code == 200


def test_chat_non_stream_usage_and_finish() -> None:
    with _client(FakeEngine()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "abc"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_chat_finish_reason_length() -> None:
    with _client(FakeEngine()) as client:
        data = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
        ).json()
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] == 2


def test_chat_stream_and_include_usage() -> None:
    with _client(FakeEngine()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.status_code == 200
        assert response.text.rstrip().endswith("[DONE]")
        events = _sse_events(response.text)
        text = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if event.get("choices")
        )
        assert text == "abc"
        assert events[-1]["usage"]["completion_tokens"] == 3
        assert any(
            event.get("choices") and event["choices"][0]["finish_reason"] == "length"
            for event in events
        )


def test_stop_sequence_truncates_and_stops() -> None:
    engine = FakeEngine(output=[b"hello ", b"ST", b"OP", b" tail"])
    with _client(engine) as client:
        data = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
                "stop": "STOP",
            },
        ).json()
        assert data["choices"][0]["message"]["content"] == "hello "
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 3


def test_completion_stream() -> None:
    with _client(FakeEngine()) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 2, "stream": True},
        )
        assert response.status_code == 200
        events = _sse_events(response.text)
        text = "".join(choice["text"] for event in events for choice in event.get("choices", []))
        assert text == "ab"
        assert any(
            choice["finish_reason"] == "length"
            for event in events
            for choice in event.get("choices", [])
        )


def test_empty_prompt_and_context_limit() -> None:
    with _client(FakeEngine()) as client:
        empty = client.post("/v1/completions", json={"prompt": "", "max_tokens": 2})
        assert empty.status_code == 400
        assert empty.json()["error"]["type"] == "invalid_request_error"
    with _client(FakeEngine(max_context=3)) as client:
        too_long = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        )
        assert too_long.status_code == 400
        assert "max_context" in too_long.json()["error"]["message"]


def test_validation_errors_are_openai_shaped() -> None:
    with _client(FakeEngine()) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "temperature": 3},
        )
        assert response.status_code == 400


def test_multimodal_chat_is_rejected() -> None:
    with _client(FakeEngine()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": "file:///tmp/x.png"},
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 400
        assert "multimodal" in response.json()["error"]["message"]


def test_stream_error_is_reported_then_done() -> None:
    engine = FakeEngine(fail_after=1)
    with _client(engine) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 3,
            },
        )
        assert response.status_code == 200
        assert "server_error" in response.text
        assert response.text.rstrip().endswith("[DONE]")


def test_engine_is_closed_on_shutdown() -> None:
    engine = FakeEngine()
    with _client(engine):
        pass
    assert engine.closed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_invalid_role_is_rejected() -> None:
    with _client(FakeEngine()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "hacker", "content": "hi"}]},
        )
        assert response.status_code == 400
