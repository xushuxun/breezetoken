"""OpenAI-compatible HTTP server for a local Qwen3.5 GGUF model.

The HTTP layer is deliberately thin: it validates OpenAI-shaped requests,
renders chat prompts with the model's own template, and streams decoded text
from a single ``Engine``.  Model work runs on a one-worker executor, which is
the serialisation point; a cancelled request sets its cancel flag and the
engine stops at the next token, so the next request can never overlap a model
mutation.

Endpoints
    GET  /                     web chat
    GET  /healthz              liveness/model summary
    GET  /v1/models
    POST /v1/chat/completions  stream and non-stream
    POST /v1/completions       stream and non-stream
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from breezetoken import __version__
from breezetoken.engine import Engine
from breezetoken.streaming import GenerationJob, IncrementalUTF8Decoder, StopBuffer

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_CHAT_MAX_TOKENS = 256
DEFAULT_COMPLETION_MAX_TOKENS = 64
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@dataclass(frozen=True)
class Settings:
    model_path: str
    backend: str = "auto"
    device: str = "auto"
    dtype: str | None = None
    max_context: int | None = None
    max_new_tokens: int = 4096
    warmup: bool = True
    host: str = "127.0.0.1"
    port: int = 8321


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "developer", "function"] = "user"
    content: str | list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)
    enable_thinking: bool = False
    tools: list[dict[str, Any]] | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    prompt: str | list[str] | None = None
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stop: str | list[str] | None = None


# --------------------------------------------------------------------------- #
# Generation plumbing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenParams:
    ids: list[int]
    max_new: int
    temperature: float
    top_p: float
    seed: int | None
    stops: tuple[str, ...]


@dataclass
class StreamState:
    tokens: int = 0
    reason: Literal["stop", "length"] | None = None


def _stop_sequences(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(item for item in value if item)


async def _iter_text(
    engine: Engine,
    executor: ThreadPoolExecutor,
    params: GenParams,
    state: StreamState,
) -> AsyncIterator[str]:
    """Yield decoded text chunks, applying stop sequences and cancellation."""
    cancel = threading.Event()
    job = GenerationJob(
        executor,
        engine.generate_stream(
            params.ids,
            params.max_new,
            params.temperature,
            top_p=params.top_p,
            seed=params.seed,
            cancel=cancel,
        ),
        cancel=cancel,
    )
    decoder = IncrementalUTF8Decoder()
    stopper = StopBuffer(params.stops)
    try:
        async for token in job:
            state.tokens += 1
            text = decoder.feed(engine.token_bytes(token))
            if not text:
                continue
            out = stopper.feed(text)
            if out:
                yield out
            if stopper.stopped:
                state.reason = "stop"
                break
        if state.reason is None:
            tail = decoder.finish()
            if tail:
                out = stopper.feed(tail)
                if out:
                    yield out
            if stopper.stops:
                held = stopper.finish()
                if held:
                    yield held
            if stopper.stopped:
                state.reason = "stop"
            else:
                state.reason = "length" if state.tokens >= params.max_new else "stop"
    finally:
        job.cancel_now()
    await job.aclose()


async def _collect_text(
    engine: Engine,
    executor: ThreadPoolExecutor,
    params: GenParams,
) -> tuple[str, StreamState]:
    state = StreamState()
    parts: list[str] = []
    async with contextlib.aclosing(_iter_text(engine, executor, params, state)) as stream:
        async for text in stream:
            parts.append(text)
    return "".join(parts), state


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _max_new(
    engine: Engine,
    prompt_len: int,
    requested: int | None,
    default: int,
    cap: int,
) -> int:
    max_new = min(requested or default, cap)
    if max_new < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be at least 1")
    if prompt_len >= engine.max_context:
        raise HTTPException(
            status_code=400,
            detail=f"prompt has {prompt_len} tokens, max_context is {engine.max_context}",
        )
    if prompt_len + max_new > engine.max_context:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt ({prompt_len}) + max_tokens ({max_new}) exceeds "
                f"max_context ({engine.max_context})"
            ),
        )
    return max_new


def _chat_chunk(
    cid: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _completion_chunk(
    cid: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}],
    }


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breezetoken-engine")
        app.state.executor = executor
        loaded = app.state.engine
        try:
            if loaded is None:
                loaded = await asyncio.to_thread(
                    Engine.load,
                    settings.model_path,
                    backend=settings.backend,
                    device=settings.device,
                    dtype=settings.dtype,
                    max_context=settings.max_context,
                )
                app.state.engine = loaded
            if settings.warmup:
                started = time.perf_counter()
                await asyncio.to_thread(loaded.warmup)
                logger.info("model warmup took %.1fs", time.perf_counter() - started)
            logger.info("serving %s", loaded)
            yield
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if loaded is not None:
                loaded.close()

    app = FastAPI(title="breezetoken", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.executor = None

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "type": error_type}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"message": message, "type": "invalid_request_error"}},
        )

    def _engine(request: Request) -> Engine:
        return request.app.state.engine

    def _executor(request: Request) -> ThreadPoolExecutor:
        return request.app.state.executor

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "chat.html").read_text(encoding="utf-8"))

    @app.get("/healthz")
    async def health(request: Request) -> dict[str, Any]:
        current = _engine(request)
        summary = getattr(current, "summary", None)
        backend = getattr(current, "backend_name", None)
        device = getattr(current, "device_name", None)
        gpu_layers = getattr(current, "gpu_layers", None)
        if summary is None:
            placement = getattr(current.model, "placement", None)
            summary = (
                placement.summary(current.model.cfg.n_layers)
                if placement is not None
                else f"{current.model.backend.name}/{current.model.device}"
            )
            backend = backend or current.model.backend.name
            device = device or current.model.device
            gpu_layers = gpu_layers if gpu_layers is not None else 0
        return {
            "status": "ok",
            "model": current.model_id,
            "backend": backend,
            "device": device,
            "placement": summary,
            "gpu_layers": gpu_layers,
            "max_context": current.max_context,
            "max_new_tokens": request.app.state.settings.max_new_tokens,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": _engine(request).model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "breezetoken",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        req: ChatCompletionRequest, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        current = _engine(request)
        executor = _executor(request)
        messages = [message.model_dump(exclude_none=True) for message in req.messages]
        try:
            ids = current.chat_ids(messages, enable_thinking=req.enable_thinking, tools=req.tools)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not ids:
            raise HTTPException(status_code=400, detail="the chat prompt tokenized to zero tokens")
        max_new = _max_new(
            current,
            len(ids),
            req.max_tokens,
            DEFAULT_CHAT_MAX_TOKENS,
            request.app.state.settings.max_new_tokens,
        )
        params = GenParams(
            ids=ids,
            max_new=max_new,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            stops=_stop_sequences(req.stop),
        )
        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        if req.stream:
            return StreamingResponse(
                _chat_sse(current, executor, params, include_usage),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )

        try:
            text, state = await _collect_text(current, executor, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - report a server error to the client
            logger.exception("chat generation failed")
            raise HTTPException(status_code=500, detail="generation failed") from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": current.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": state.reason,
                }
            ],
            "usage": _usage(len(ids), state.tokens),
        }

    async def _chat_sse(
        current: Engine,
        executor: ThreadPoolExecutor,
        params: GenParams,
        include_usage: bool,
    ) -> AsyncIterator[str]:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        state = StreamState()
        # Send a keep-alive and the role before prefill so a long prompt still
        # produces bytes immediately (and clients never see a role-less stream).
        yield ": keep-alive\n\n"
        yield _sse(
            _chat_chunk(cid, created, current.model_id, {"role": "assistant", "content": ""})
        )
        try:
            async with contextlib.aclosing(_iter_text(current, executor, params, state)) as stream:
                async for text in stream:
                    yield _sse(_chat_chunk(cid, created, current.model_id, {"content": text}))
            yield _sse(_chat_chunk(cid, created, current.model_id, {}, state.reason or "stop"))
            if include_usage:
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": current.model_id,
                        "choices": [],
                        "usage": _usage(len(params.ids), state.tokens),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - stream an error frame
            logger.exception("chat stream failed")
            yield _sse({"error": {"message": str(exc), "type": "server_error"}})
        yield "data: [DONE]\n\n"

    @app.post("/v1/completions", response_model=None)
    async def completions(
        req: CompletionRequest, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        current = _engine(request)
        executor = _executor(request)
        prompt = req.prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise HTTPException(
                    status_code=400, detail="only one prompt per request is supported"
                )
            prompt = prompt[0]
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt is required")
        ids = current.encode(prompt)
        if not ids:
            raise HTTPException(status_code=400, detail="prompt tokenized to zero tokens")
        max_new = _max_new(
            current,
            len(ids),
            req.max_tokens,
            DEFAULT_COMPLETION_MAX_TOKENS,
            request.app.state.settings.max_new_tokens,
        )
        params = GenParams(
            ids=ids,
            max_new=max_new,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            stops=_stop_sequences(req.stop),
        )

        if req.stream:
            return StreamingResponse(
                _completion_sse(current, executor, params),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )

        try:
            text, state = await _collect_text(current, executor, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("completion generation failed")
            raise HTTPException(status_code=500, detail="generation failed") from exc
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": current.model_id,
            "choices": [
                {"index": 0, "text": text, "finish_reason": state.reason, "logprobs": None}
            ],
            "usage": _usage(len(ids), state.tokens),
        }

    async def _completion_sse(
        current: Engine, executor: ThreadPoolExecutor, params: GenParams
    ) -> AsyncIterator[str]:
        cid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        state = StreamState()
        yield ": keep-alive\n\n"
        try:
            async with contextlib.aclosing(_iter_text(current, executor, params, state)) as stream:
                async for text in stream:
                    yield _sse(_completion_chunk(cid, created, current.model_id, text))
            yield _sse(
                _completion_chunk(cid, created, current.model_id, "", state.reason or "stop")
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("completion stream failed")
            yield _sse({"error": {"message": str(exc), "type": "server_error"}})
        yield "data: [DONE]\n\n"

    return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breezetoken-server",
        description="OpenAI-compatible server for a local Qwen3.5 GGUF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BREEZETOKEN_MODEL"),
        help="path to a Qwen3.5 GGUF file (or set BREEZETOKEN_MODEL)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "tilelang"],
        default="auto",
        help="auto (GPU tilelang + CPU torch placement); explicit values are debug overrides",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (hybrid placement); explicit cpu/cuda overrides disable the planner",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (or set BREEZETOKEN_MODEL)")
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    settings = Settings(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_context=args.max_context,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        host=args.host,
        port=args.port,
    )
    app = create_app(settings)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
