"""Phase-1 dequantization tests: naive ggml ports vs the vectorized decoder.

The naive reference below is a literal line-by-line port of
``dequantize_row_*`` from ggml-quants.c (MIT); the vectorized implementation
must match it exactly on hand-built blocks (which exercise signs, high bits,
scales and mins) and on random blocks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from breezetoken.quant._tables import IQ2S_GRID, IQ3S_GRID, IQ4NL_VALUES
from breezetoken.quant.dequant import dequantize
from breezetoken.quant.formats import FORMATS, GGMLType

KMASK = [1, 2, 4, 8, 16, 32, 64, 128]


def _f16(x: float) -> bytes:
    return np.float16(x).tobytes()


# --------------------------------------------------------------------------- #
# naive references (ported from ggml-quants.c)
# --------------------------------------------------------------------------- #


def ref_q8_0(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[0:2], dtype="<f2")[0])
    q = np.frombuffer(block[2:34], dtype="<i1")
    return (d * q.astype(np.float32)).tolist()


def ref_q2_k(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[80:82], dtype="<f2")[0])
    dmin = float(np.frombuffer(block[82:84], dtype="<f2")[0])
    scales = block[0:16]
    q = block[16:80]
    y: list[float] = []
    is_ = 0
    for n in range(0, 256, 128):
        qb = n // 4  # q advances 32 bytes per 128-element half
        shift = 0
        for _j in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            y += [dl * ((q[qb + sb] >> shift) & 3) - ml for sb in range(16)]
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            y += [dl * ((q[qb + sb + 16] >> shift) & 3) - ml for sb in range(16)]
            shift += 2
    return y


def _get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ref_q4_k(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[0:2], dtype="<f2")[0])
    dmin = float(np.frombuffer(block[2:4], dtype="<f2")[0])
    scales = block[4:16]
    q = block[16:144]
    y: list[float] = []
    is_ = 0
    for j in range(0, 256, 64):
        qb = (j // 64) * 32  # qs advances 32 bytes per 64-element group
        sc, m = _get_scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        y += [d1 * (q[qb + sb] & 0xF) - m1 for sb in range(32)]
        y += [d2 * (q[qb + sb] >> 4) - m2 for sb in range(32)]
        is_ += 2
    return y


def ref_q6_k(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[208:210], dtype="<f2")[0])
    ql, qh, sc = block[0:128], block[128:192], np.frombuffer(block[192:208], dtype="<i1")
    y = [0.0] * 256
    for n in range(0, 256, 128):
        hb = n // 128
        qlb, qhb, scb = 64 * hb, 32 * hb, 8 * hb  # ql/qh/sc advance per half
        for sb in range(32):
            is_ = sb // 16
            q1 = ((ql[qlb + sb] & 0xF) | ((qh[qhb + sb] & 3) << 4)) - 32
            q2 = ((ql[qlb + sb + 32] & 0xF) | (((qh[qhb + sb] >> 2) & 3) << 4)) - 32
            q3 = ((ql[qlb + sb] >> 4) | (((qh[qhb + sb] >> 4) & 3) << 4)) - 32
            q4 = ((ql[qlb + sb + 32] >> 4) | (((qh[qhb + sb] >> 6) & 3) << 4)) - 32
            y[n + sb] = d * sc[scb + is_] * q1
            y[n + sb + 32] = d * sc[scb + is_ + 2] * q2
            y[n + sb + 64] = d * sc[scb + is_ + 4] * q3
            y[n + sb + 96] = d * sc[scb + is_ + 6] * q4
    return y


def ref_iq2_s(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[0:2], dtype="<f2")[0])
    qs, qh, scales = block[2:66], block[66:74], block[74:82]
    signs = qs[32:]
    y: list[float] = []
    for ib32 in range(8):
        db0 = d * (0.5 + (scales[ib32] & 0xF)) * 0.25
        db1 = d * (0.5 + (scales[ib32] >> 4)) * 0.25
        for sb in range(4):
            dl = db0 if sb < 2 else db1
            idx = qs[4 * ib32 + sb] | ((qh[ib32] << (8 - 2 * sb)) & 0x300)
            grid = IQ2S_GRID[idx]
            y += [
                float(dl) * int(grid[j]) * (-1 if signs[4 * ib32 + sb] & KMASK[j] else 1)
                for j in range(8)
            ]
    return y


def ref_iq3_s(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[0:2], dtype="<f2")[0])
    qs, qh, signs, scales = block[2:66], block[66:74], block[74:106], block[106:110]
    y: list[float] = []
    for ib32 in range(0, 8, 2):
        db1 = d * (1 + 2 * (scales[ib32 // 2] & 0xF))
        db2 = d * (1 + 2 * (scales[ib32 // 2] >> 4))
        for pair, db in ((0, db1), (1, db2)):
            qsh = qh[ib32 + pair]
            for sb in range(4):
                base = 16 * (ib32 // 2) + 8 * pair + 2 * sb
                g1 = IQ3S_GRID[qs[base] | ((qsh << (8 - 2 * sb)) & 256)]
                g2 = IQ3S_GRID[qs[base + 1] | ((qsh << (7 - 2 * sb)) & 256)]
                sbyte = signs[8 * (ib32 // 2) + 4 * pair + sb]
                for j in range(4):
                    y.append(float(db) * int(g1[j]) * (-1 if sbyte & KMASK[j] else 1))
                for j in range(4):
                    y.append(float(db) * int(g2[j]) * (-1 if sbyte & KMASK[j + 4] else 1))
    return y


def ref_iq4_xs(block: bytes) -> list[float]:
    d = float(np.frombuffer(block[0:2], dtype="<f2")[0])
    scales_h = int.from_bytes(block[2:4], "little")
    scales_l = block[4:8]
    qs = block[8:136]
    y: list[float] = []
    for ib in range(8):
        ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        for j in range(16):
            y.append(dl * int(IQ4NL_VALUES[qs[16 * ib + j] & 0xF]))
        for j in range(16):
            y.append(dl * int(IQ4NL_VALUES[qs[16 * ib + j] >> 4]))
    return y


REFS = {
    "q8_0": (GGMLType.Q8_0, 32, ref_q8_0),
    "q2_K": (GGMLType.Q2_K, 256, ref_q2_k),
    "q4_K": (GGMLType.Q4_K, 256, ref_q4_k),
    "q6_K": (GGMLType.Q6_K, 256, ref_q6_k),
    "iq2_s": (GGMLType.IQ2_S, 256, ref_iq2_s),
    "iq3_s": (GGMLType.IQ3_S, 256, ref_iq3_s),
    "iq4_xs": (GGMLType.IQ4_XS, 256, ref_iq4_xs),
}


# --------------------------------------------------------------------------- #
# block builders
# --------------------------------------------------------------------------- #


def make_q8_0(rng: np.random.Generator) -> bytes:
    return (
        _f16(rng.uniform(0.01, 2.0))
        + rng.integers(-127, 128, 32, dtype=np.int16).astype(np.int8).tobytes()
    )


def make_q2_k(rng) -> bytes:
    scales = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    q = rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
    return scales + q + _f16(rng.uniform(0.01, 2.0)) + _f16(rng.uniform(0.01, 2.0))


def make_q4_k(rng) -> bytes:
    return (
        _f16(rng.uniform(0.01, 2.0))
        + _f16(rng.uniform(0.01, 2.0))
        + rng.integers(0, 256, 12, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    )


def make_q6_k(rng) -> bytes:
    return (
        rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        + rng.integers(-128, 128, 16, dtype=np.int16).astype(np.int8).tobytes()
        + _f16(rng.uniform(0.01, 2.0))
    )


def make_iq2_s(rng) -> bytes:
    return (
        _f16(rng.uniform(0.01, 2.0))
        + rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
    )


def make_iq3_s(rng) -> bytes:
    return (
        _f16(rng.uniform(0.01, 2.0))
        + rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 4, dtype=np.uint8).tobytes()
    )


def make_iq4_xs(rng) -> bytes:
    return (
        _f16(rng.uniform(0.01, 2.0))
        + rng.integers(0, 65536, 1, dtype=np.uint16).tobytes()
        + rng.integers(0, 256, 4, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    )


MAKERS = {
    "q8_0": make_q8_0,
    "q2_K": make_q2_k,
    "q4_K": make_q4_k,
    "q6_K": make_q6_k,
    "iq2_s": make_iq2_s,
    "iq3_s": make_iq3_s,
    "iq4_xs": make_iq4_xs,
}


@pytest.mark.parametrize("name", list(REFS))
def test_block_formats_match_naive_reference(name: str) -> None:
    type_id, columns, ref = REFS[name]
    fmt = FORMATS[type_id]
    rng = np.random.default_rng(1234)
    for _ in range(8):
        block = MAKERS[name](rng)
        assert len(block) == fmt.bytes_per_block
        got = dequantize(torch.from_numpy(np.frombuffer(block, dtype=np.uint8)), fmt, (columns,))
        want = torch.tensor(ref(block), dtype=torch.float32)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fixed_vector_q8_0() -> None:
    fmt = FORMATS[GGMLType.Q8_0]
    q = np.arange(-16, 16, dtype=np.int8)
    block = _f16(2.0) + q.tobytes()
    got = dequantize(torch.from_numpy(np.frombuffer(block, dtype=np.uint8)), fmt, (32,))
    torch.testing.assert_close(got, torch.tensor((2.0 * q).tolist(), dtype=torch.float32))


def test_fixed_vector_iq3_s_signs_and_high_bits() -> None:
    """Hand-computed iq3_s block: d=1, first pair scale nibbles 0/1."""
    fmt = FORMATS[GGMLType.IQ3_S]
    d = _f16(1.0)
    qs = bytes([0, 0, 0, 0, 0, 0, 0, 0] * 8)  # grid row 0 everywhere -> [1,1,1,1]
    qh = bytes(8)  # no high bits
    signs = bytes(32)  # all positive
    scales = bytes([0x10, 0, 0, 0])  # pair 0: db1 = 1, db2 = 3
    block = d + qs + qh + signs + scales
    got = dequantize(torch.from_numpy(np.frombuffer(block, dtype=np.uint8)), fmt, (256,))
    want = torch.tensor([1.0] * 32 + [3.0] * 32 + [1.0] * 192)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fixed_vector_iq4_xs() -> None:
    fmt = FORMATS[GGMLType.IQ4_XS]
    d = _f16(1.0)
    scales_h = (1).to_bytes(2, "little")  # ib=0: high bits 01 -> ls = 16 | nibble
    scales_l = bytes(4)  # low nibbles 0 -> ls = 16 -> dl = -16; other groups ls=0 -> dl=0
    qs = bytes([0x10, 0x32] + [0] * 126)  # byte0: nibs (0,1); byte1: nibs (2,3)
    block = d + scales_h + scales_l + qs
    got = dequantize(torch.from_numpy(np.frombuffer(block, dtype=np.uint8)), fmt, (256,))
    lo = [-16.0 * int(v) for v in IQ4NL_VALUES]
    want = [lo[0], lo[2]] + [lo[0]] * 14 + [lo[1], lo[3]] + [lo[0]] * 14
    want += [4064.0] * (256 - 32)  # ls=0 elsewhere -> dl=-32 with zero qs
    torch.testing.assert_close(got, torch.tensor(want), rtol=0, atol=0)


def test_f32_and_bf16() -> None:
    f32 = FORMATS[GGMLType.F32]
    bf16 = FORMATS[GGMLType.BF16]
    x = torch.randn(4, 64, dtype=torch.float32)
    got = dequantize(
        torch.from_numpy(np.frombuffer(x.numpy().tobytes(), dtype=np.uint8)), f32, (4, 64)
    )
    torch.testing.assert_close(got, x)
    got_bf = dequantize(
        torch.from_numpy(
            np.frombuffer(x.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), dtype=np.uint8)
        ),
        bf16,
        (4, 64),
    )
    torch.testing.assert_close(got_bf, x.to(torch.bfloat16).float())
