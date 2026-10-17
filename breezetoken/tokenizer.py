"""GGUF tokenizer and chat prompt builder.

The tokenization itself is delegated to HuggingFace's Rust ``tokenizers``
runtime.  The hand-written Python BPE that used to live here could not match
the model's tokenizer.json for NFC-normalized text, combining marks, emoji
variation selectors, or special tokens; using the same runtime and the same
pre-tokenizer regex removes that whole class of bug.

The chat template comes from GGUF metadata (``tokenizer.chat_template``) and is
rendered with a sandboxed Jinja2 environment, so the prompt the server sends is
the prompt the model was trained with, including the empty ``<think>`` block
that Qwen3.5 expects when thinking is disabled.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)

from breezetoken.gguf import GGUFFile

logger = logging.getLogger(__name__)

# The Qwen3.5 pre-tokenizer, copied verbatim from the model's tokenizer.json.
# Differences from GPT-2 matter: combining marks belong to the word, and
# numbers are split one digit at a time.
_QWEN_PRE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_PRE_PATTERNS = {
    "qwen35": _QWEN_PRE,
    "qwen3": _QWEN_PRE,
    "qwen2": _QWEN_PRE,
}

# Fallback for GGUFs that predate chat_template metadata.
_FALLBACK_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is true %}"
    "{{- '<think>\\n' }}"
    "{%- else %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)

_CONTROL, _USER_DEFINED = 3, 4


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


_BYTE_TO_CHAR = _bytes_to_unicode()
_CHAR_TO_BYTE = {char: byte for byte, char in _BYTE_TO_CHAR.items()}


def _raise_exception(message: str) -> None:
    raise ValueError(message)


def _is_text_only_content(content: Any) -> bool:
    if content is None or isinstance(content, str):
        return True
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return False
    for item in content:
        if not isinstance(item, Mapping):
            return False
        kind = item.get("type")
        if "image" in item or "image_url" in item or kind == "image":
            return False
        if "video" in item or kind == "video":
            return False
        if kind not in (None, "text"):
            return False
    return True


class BPETokenizer:
    """A tokenizer built from GGUF metadata, matching the HF tokenizer.json."""

    def __init__(self, gguf: GGUFFile) -> None:
        md = gguf.metadata
        self.tokens: list[str] = list(md["tokenizer.ggml.tokens"])
        self.token_types: list[int] = [int(t) for t in md["tokenizer.ggml.token_type"]]
        if len(self.tokens) != len(self.token_types):
            raise ValueError("GGUF token/token_type lengths differ")

        self.vocab: dict[str, int] = {token: i for i, token in enumerate(self.tokens)}
        self.bos_id = int(md.get("tokenizer.ggml.bos_token_id", -1))
        self.eos_id = int(md["tokenizer.ggml.eos_token_id"])
        self.pad_id = int(md.get("tokenizer.ggml.padding_token_id", -1))
        self.add_bos = bool(md.get("tokenizer.ggml.add_bos_token", False))
        self.pre = str(md.get("tokenizer.ggml.pre", "qwen35"))

        self._special_ids = frozenset(
            i for i, kind in enumerate(self.token_types) if kind in (_CONTROL, _USER_DEFINED)
        )
        self._special_texts = tuple(self.tokens[i] for i in sorted(self._special_ids))
        self._eos_ids = {self.eos_id}
        for name in ("<|im_end|>", "<|endoftext|>"):
            if name in self.vocab:
                self._eos_ids.add(self.vocab[name])

        self._tokenizer = self._build_tokenizer(md)
        template = md.get("tokenizer.chat_template")
        self._template = self._build_template(str(template) if template else _FALLBACK_TEMPLATE)
        self.has_chat_template = bool(template)

    # ------------------------------------------------------------------ build
    def _build_tokenizer(self, md: Mapping[str, Any]) -> Tokenizer:
        merges: list[tuple[str, str]] = []
        skipped = 0
        for merge in md["tokenizer.ggml.merges"]:
            parts = str(merge).split(" ", 1)
            if len(parts) != 2:
                skipped += 1
                continue
            left, right = parts
            if left in self.vocab and right in self.vocab and left + right in self.vocab:
                merges.append((left, right))
            else:
                skipped += 1
        if skipped:
            logger.warning("skipped %d unusable BPE merges", skipped)

        tokenizer = Tokenizer(
            models.BPE(vocab=self.vocab, merges=merges, unk_token=None, fuse_unk=False)
        )
        tokenizer.normalizer = normalizers.NFC()
        pre = _PRE_PATTERNS.get(self.pre)
        if pre is None:
            logger.warning("unknown GGUF pre-tokenizer %r; using the Qwen pattern", self.pre)
            pre = _QWEN_PRE
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(pre), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False, trim_offsets=False, use_regex=False
                ),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(
            add_prefix_space=False, trim_offsets=False, use_regex=False
        )

        specials = [
            AddedToken(self.tokens[i], normalized=False, special=True)
            for i in sorted(self._special_ids)
        ]
        if specials:
            tokenizer.add_special_tokens(specials)
        return tokenizer

    @staticmethod
    def _build_template(template: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise_exception
        return env.from_string(template)

    # ------------------------------------------------------------------- text
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def special_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def special_texts(self) -> tuple[str, ...]:
        return self._special_texts

    @property
    def eos_ids(self) -> frozenset[int]:
        return frozenset(self._eos_ids)

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_ids

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        add_bos: bool | None = None,
    ) -> list[int]:
        ids = list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)
        if (add_bos if add_bos is not None else self.add_bos) and self.bos_id >= 0:
            ids.insert(0, self.bos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        clean = [i for i in ids if 0 <= i < self.vocab_size]
        if not clean:
            return ""
        return self._tokenizer.decode(clean, skip_special_tokens=skip_special_tokens)

    def token_bytes(self, token_id: int, *, include_special: bool = False) -> bytes:
        """The raw UTF-8 bytes a token contributes to the stream.

        Special tokens contribute nothing by default: they are prompt/control
        markers, and HF's decoder skips them too.
        """
        if token_id < 0 or token_id >= self.vocab_size:
            return b""
        if token_id in self._special_ids and not include_special:
            return b""
        token = self._tokenizer.id_to_token(token_id)
        if not token:
            return b""
        out = bytearray()
        for char in token:
            byte = _CHAR_TO_BYTE.get(char)
            if byte is None:
                out += char.encode("utf-8")
            else:
                out.append(byte)
        return bytes(out)

    # ------------------------------------------------------------------- chat
    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_vision_id: bool = False,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        for message in messages:
            content = message.get("content")
            if not _is_text_only_content(content):
                raise ValueError("multimodal content is not supported by the qwen35 text engine")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
                texts.extend(
                    item["text"]
                    for item in content
                    if isinstance(item, Mapping) and isinstance(item.get("text"), str)
                )
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                texts.append(reasoning)
            for text in texts:
                for reserved in self._special_texts:
                    if reserved in text:
                        raise ValueError(
                            f"message content contains reserved special token {reserved!r}"
                        )
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            add_vision_id=add_vision_id,
        )

    def chat_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[int]:
        text = self.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        return self.encode(text, add_bos=False)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab={self.vocab_size}, special={len(self._special_ids)}, "
            f"pre={self.pre!r}, template={self.has_chat_template})"
        )
