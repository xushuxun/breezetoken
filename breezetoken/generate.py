"""Command-line generation with the breezetoken qwen35 engine.

    breezetoken --model <gguf> --prompt "你好"
    breezetoken --model <gguf> --prompt "..." --backend tilelang --device cuda

The prompt is rendered through the model's own chat template by default; pass
``--raw`` to feed it as a completion prefix instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence

from breezetoken.engine import Engine
from breezetoken.streaming import IncrementalUTF8Decoder

_DEFAULT_MODEL_ENV = "BREEZETOKEN_MODEL"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breezetoken",
        description="Local Qwen3.5 inference (GGUF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(_DEFAULT_MODEL_ENV),
        help=f"path to a Qwen3.5 GGUF file (or set {_DEFAULT_MODEL_ENV})",
    )
    parser.add_argument("--prompt", help="prompt text; reads stdin when omitted")
    parser.add_argument("--system", default=None, help="system message for chat mode")
    parser.add_argument("--raw", action="store_true", help="do not apply the chat template")
    parser.add_argument("--max-new-tokens", "--n", dest="max_new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", choices=["auto", "torch", "tilelang"], default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype", default=None, help="auto, float32, bfloat16 (tilelang CUDA: bfloat16)"
    )
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def _prompt_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.prompt is not None:
        return args.prompt
    if sys.stdin.isatty():
        parser.error("--prompt is required when stdin is a terminal")
    return sys.stdin.read().strip()


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error(f"--model is required (or set {_DEFAULT_MODEL_ENV})")
    prompt = _prompt_from_args(args, parser)

    started = time.perf_counter()
    engine: Engine | None = None
    try:
        engine = Engine.load(
            args.model,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            max_context=args.max_context,
        )
        load_seconds = time.perf_counter() - started
        if not args.no_warmup:
            engine.warmup()
        ids = engine.encode(prompt) if args.raw else engine.chat_ids(_messages(prompt, args.system))
        if not ids:
            raise ValueError("the prompt tokenized to zero tokens")

        decoder = IncrementalUTF8Decoder()
        token_count = 0
        first_token_at: float | None = None
        gen_started = time.perf_counter()
        try:
            for token in engine.generate_stream(
                ids,
                args.max_new,
                args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                text = decoder.feed(engine.token_bytes(token))
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                token_count += 1
            tail = decoder.finish()
            if tail:
                sys.stdout.write(tail)
        finally:
            sys.stdout.flush()

        total = time.perf_counter() - gen_started
        ttft = (first_token_at - gen_started) if first_token_at is not None else total
        rate = token_count / total if total > 0 else 0.0
        print(
            f"\n[{engine.model_id} | {engine.model.backend.name}/{engine.model.device} | "
            f"load {load_seconds:.1f}s | prefill+first {ttft * 1e3:.0f}ms | "
            f"{token_count} tokens, {rate:.2f} tok/s]",
            file=sys.stderr,
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
