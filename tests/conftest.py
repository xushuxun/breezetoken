"""Shared fixtures for the test suite.

The tokenizer/GGUF/model tests need the local Qwen3.5 GGUF; they are skipped
when it is absent.  Unit tests (streaming, HTTP shaping) run everywhere.

tilelang writes compiled kernels to ``$TILELANG_CACHE_DIR`` (default
``~/.tilelang``); point it at a temporary directory before tilelang is imported
so tests never depend on a writable home directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "TILELANG_CACHE_DIR", str(Path(tempfile.gettempdir()) / "breezetoken-tilelang-cache")
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402

_DEFAULT_MODEL = (
    Path.home() / ".cache/modelscope/models/unsloth--Qwen3.5-0.8B-MTP-GGUF/snapshots/master/"
    "Qwen3.5-0.8B-Q8_0.gguf"
)
_DEFAULT_HF = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots"


def _find_hf_tokenizer() -> Path | None:
    candidate = Path(os.environ.get("BREEZETOKEN_TEST_HF", _DEFAULT_HF))
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        direct = candidate / "tokenizer.json"
        if direct.is_file():
            return direct
        snapshots = sorted(candidate.glob("*/tokenizer.json"))
        if snapshots:
            return snapshots[-1]
    return None


@pytest.fixture(scope="session")
def model_path() -> Path:
    path = Path(os.environ.get("BREEZETOKEN_TEST_MODEL", _DEFAULT_MODEL))
    if not path.is_file():
        pytest.skip(f"Qwen3.5 GGUF not found at {path}; set BREEZETOKEN_TEST_MODEL")
    return path


@pytest.fixture(scope="session")
def gguf(model_path: Path):
    from breezetoken.gguf import GGUFFile

    file = GGUFFile(model_path)
    try:
        yield file
    finally:
        file.close()


@pytest.fixture(scope="session")
def tokenizer(gguf):
    from breezetoken.tokenizer import BPETokenizer

    return BPETokenizer(gguf)


@pytest.fixture(scope="session")
def hf_tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    path = _find_hf_tokenizer()
    if path is None:
        pytest.skip("HF tokenizer.json not found; set BREEZETOKEN_TEST_HF")
    return AutoTokenizer.from_pretrained(str(path.parent if path.is_file() else path))


@pytest.fixture(scope="session")
def torch_model(gguf):
    from breezetoken.model.qwen35 import Qwen35

    return Qwen35(gguf, backend="torch", device="cpu")
