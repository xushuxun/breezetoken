"""The runtime layer: model + tokenizer + generation policy.

The server and CLI only talk to ``Engine``.  Chat formatting, EOS handling,
context limits, and cancellation live here so the HTTP layer stays a thin
translation between OpenAI's schema and this API.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breezetoken.gguf import GGUFFile
from breezetoken.model.qwen35 import Qwen35
from breezetoken.tokenizer import BPETokenizer


@dataclass
class Engine:
    model: Qwen35
    tokenizer: BPETokenizer
    model_id: str

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        backend: str = "auto",
        device: str = "auto",
        dtype: str | None = None,
        max_context: int | None = None,
    ) -> Engine:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        with GGUFFile(path) as gguf:
            arch = gguf.arch
            native_quant = all(
                gguf.tensor_info(name)[1] in ("f32", "q8_0") for name in gguf.tensor_names()
            )
            if arch != "qwen35" or not native_quant:
                raise NotImplementedError(
                    f"{arch} / non-Q8 GGUF is not supported by the native engine yet; "
                    "see docs/qwen36_35b_plan.md for the native IQ3_S/MoE plan"
                )
            tokenizer = BPETokenizer(gguf)
            model = Qwen35(
                gguf,
                backend=backend,
                device=device,
                dtype=dtype,
                max_context=max_context,
            )
            model_id = str(gguf.metadata.get("general.name") or path.stem)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id)

    # ------------------------------------------------------------------ text
    @property
    def max_context(self) -> int:
        return self.model.cfg.max_context

    @property
    def eos_ids(self) -> frozenset[int]:
        return self.tokenizer.eos_ids

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[int]:
        return self.tokenizer.chat_ids(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
        )

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_bytes(self, token_id: int) -> bytes:
        return self.tokenizer.token_bytes(token_id)

    # -------------------------------------------------------------- generate
    def generate_stream(
        self,
        ids: Sequence[int],
        max_new: int,
        temperature: float = 0.0,
        *,
        top_p: float = 1.0,
        seed: int | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[int]:
        yield from self.model.generate_stream(
            ids,
            max_new,
            temperature,
            top_p=top_p,
            eos_ids=self.eos_ids,
            cancel=cancel,
            seed=seed,
        )

    def warmup(self) -> None:
        self.model.warmup()

    def close(self) -> None:
        self.model.close()

    @property
    def summary(self) -> str:
        return self.model.placement.summary(self.model.cfg.n_layers)

    @property
    def backend_name(self) -> str:
        return self.model.backend.name

    @property
    def device_name(self) -> str:
        return self.model.device

    @property
    def gpu_layers(self) -> int:
        return self.model.placement.gpu_layers

    def __repr__(self) -> str:
        return (
            f"Engine(model={self.model_id!r}, placement={self.summary}, "
            f"max_context={self.max_context})"
        )
