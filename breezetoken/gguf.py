"""Minimal GGUF v3 reader: metadata, tensor views, and Q8_0 dequantization.

The file is memory-mapped and tensors are materialized on demand.  Numeric
metadata arrays are copied out of the mapping so the file can be closed after
the tokenizer/model have loaded -- a memmap view outliving its mapping is a
segfault waiting to happen.

Two format traps encoded here:

* the default tensor-data alignment is 32 bytes, not 64;
* dimensions are stored fastest-first, so the logical shape is reversed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

GGML_F32: Final = 0
GGML_Q8_0: Final = 8

_TYPE_NAMES: Final = {GGML_F32: "f32", GGML_Q8_0: "q8_0"}

_VAL: Final = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}

#: Bytes per row chunk during dequantization: bounds the temporary f32 copy.
_CHUNK_BYTES = 64 << 20


class GGUFFile:
    """A read-only GGUF file with metadata and lazily materialized tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._buf: np.memmap | None = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.metadata: dict[str, Any] = {}
        self._tensors: dict[str, tuple[tuple[int, ...], int, int]] = {}
        self._parse()

    # ------------------------------------------------------------------ parse
    def _read_value(self, pos: int, vtype: int) -> tuple[Any, int]:
        buf = self._require_buf()
        if vtype == 8:  # string
            (n,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            return bytes(buf[pos : pos + n]).decode("utf-8"), pos + n
        if vtype == 9:  # array
            (etype, count) = struct.unpack_from("<IQ", buf, pos)
            pos += 12
            if etype == 8:
                items = []
                for _ in range(count):
                    item, pos = self._read_value(pos, 8)
                    items.append(item)
                return items, pos
            fmt, _ = _VAL[etype]
            itemsize = struct.calcsize("<" + fmt)
            # .copy() keeps metadata usable after the mmap is closed.
            arr = np.frombuffer(buf, dtype="<" + fmt, count=count, offset=pos).copy()
            return arr, pos + itemsize * count
        fmt, size = _VAL[vtype]
        (value,) = struct.unpack_from("<" + fmt, buf, pos)
        return value, pos + size

    def _parse(self) -> None:
        buf = self._require_buf()
        if bytes(buf[:4]) != b"GGUF":
            raise ValueError(f"{self.path} is not a GGUF file")
        (version, n_tensors, n_kv) = struct.unpack_from("<IQQ", buf, 4)
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")

        pos = 24
        for _ in range(n_kv):
            (klen,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            key = bytes(buf[pos : pos + klen]).decode("utf-8")
            pos += klen
            (vtype,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            value, pos = self._read_value(pos, vtype)
            self.metadata[key] = value

        alignment = int(self.metadata.get("general.alignment", 32))
        infos: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            (nlen,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            name = bytes(buf[pos : pos + nlen]).decode("utf-8")
            pos += nlen
            (ndim,) = struct.unpack_from("<I", buf, pos)
            dims = struct.unpack_from(f"<{ndim}Q", buf, pos + 4)
            pos += 4 + 8 * ndim
            (ttype,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            (toff,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            infos.append((name, tuple(reversed(dims)), ttype, toff))

        data_start = (pos + alignment - 1) // alignment * alignment
        for name, shape, ttype, toff in infos:
            self._tensors[name] = (shape, ttype, data_start + toff)

    # ------------------------------------------------------------------ query
    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def tensor_info(self, name: str) -> tuple[tuple[int, ...], str]:
        try:
            shape, ttype, _ = self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor {name!r} not found in {self.path.name}") from exc
        return shape, _TYPE_NAMES.get(ttype, f"type_{ttype}")

    @property
    def arch(self) -> str:
        return str(self.metadata["general.architecture"])

    # ------------------------------------------------------------------ load
    def get_tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialize a tensor as a row-major torch tensor."""
        try:
            shape, ttype, offset = self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor {name!r} not found in {self.path.name}") from exc
        if ttype == GGML_F32:
            return self._read_f32(shape, offset, dtype)
        if ttype == GGML_Q8_0:
            return self._dequant_q8_0(shape, offset, dtype)
        raise NotImplementedError(f"ggml type {ttype} is not supported for {name}")

    def _read_f32(self, shape: tuple[int, ...], offset: int, dtype: torch.dtype) -> torch.Tensor:
        if len(shape) != 2:
            # Small 1-D / 4-D tensors are copied in one piece; only large 2-D
            # weights need chunking to keep peak RSS bounded.
            arr = np.frombuffer(
                self._require_buf(), dtype="<f4", count=int(np.prod(shape)), offset=offset
            )
            return torch.from_numpy(arr.copy()).reshape(shape).to(dtype)
        rows, cols = shape
        out = torch.empty(shape, dtype=dtype)
        row_bytes = max(1, _CHUNK_BYTES // max(1, cols * 4))
        for start in range(0, rows, row_bytes):
            stop = min(rows, start + row_bytes)
            arr = np.frombuffer(
                self._require_buf(),
                dtype="<f4",
                count=(stop - start) * cols,
                offset=offset + start * cols * 4,
            )
            out[start:stop].copy_(torch.from_numpy(arr.copy()).reshape(stop - start, cols))
        return out

    def _dequant_q8_0(
        self, shape: tuple[int, ...], offset: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if len(shape) != 2:
            raise NotImplementedError(f"Q8_0 only for 2-D tensors, got {shape}")
        rows, cols = shape
        if cols % 32 != 0:
            raise ValueError(f"Q8_0 inner dim must be a multiple of 32, got {shape}")
        blocks_per_row = cols // 32
        block = np.dtype([("scale", "<f2"), ("q", "<i1", (32,))])
        block_bytes = block.itemsize
        if block_bytes != 34:
            raise RuntimeError(f"unexpected Q8_0 block size {block_bytes}")

        out = torch.empty(shape, dtype=dtype)
        row_chunk = max(1, _CHUNK_BYTES // max(1, cols * 4))
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            raw = np.frombuffer(
                self._require_buf(),
                dtype=block,
                count=(stop - start) * blocks_per_row,
                offset=offset + start * blocks_per_row * block_bytes,
            )
            q = raw["q"].astype(np.float32)
            scale = raw["scale"].astype(np.float32)
            w = q * scale[:, None]
            out[start:stop].copy_(torch.from_numpy(w.reshape(stop - start, cols)))
        return out

    # ------------------------------------------------------------- lifecycle
    def _require_buf(self) -> np.memmap:
        if self._buf is None:
            raise ValueError(f"GGUF file {self.path} is closed")
        return self._buf

    def close(self) -> None:
        buf, self._buf = self._buf, None
        if buf is not None:
            mmap = getattr(buf, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> GGUFFile:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._buf is None else f"tensors={len(self._tensors)}"
        return f"GGUFFile({self.path.name}, {state})"
