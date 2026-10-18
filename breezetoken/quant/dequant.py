"""Vectorized reference dequantization for the native GGUF quant path.

Every format follows the same contract: a row of ``columns`` elements is a
sequence of fixed-size blocks whose layout matches ``ggml-common.h``; blocks
run along the last dimension.  The implementations below are direct ports of
the ``dequantize_row_*`` functions in ``ggml-quants.c`` (MIT), written with
torch elementwise ops so the same code runs on CPU and CUDA without
materializing more than the requested row tile.

The math per format (d = fp16 scale, signs use bit j of the sign byte):

* q8_0:    y = d * q
* q2_K:    y = d*(sc&0xF)*((q>>(2j))&3) - dmin*(sc>>4), 16B scales/64B qs
* q4_K:    y = d*sc*(q nib) - dmin*m, scales via get_scale_min_k4
* q6_K:    y = d*sc[8h+is+2k]*q, is=l//16, q rebuilt from ql/qh minus 32
* iq2_s:   y = d*(0.5+nib)*0.25*grid[idx]*(+-1), idx = qs | 2 qh bits
* iq3_s:   y = d*(1+2*nib)*grid[idx]*(+-1),     idx = qs | 1 qh bit
* iq4_xs:  y = d*(ls-32)*kvalues_iq4nl[qs nib]
"""

from __future__ import annotations

import torch

from breezetoken.quant._tables import IQ2S_GRID, IQ3S_GRID, IQ4NL_VALUES, IQ_SIGN_MASK
from breezetoken.quant.formats import GGMLType, QuantFormat

_TABLES: dict[str, torch.Tensor | None] = {}


def _table(name: str, values, device: torch.device) -> torch.Tensor:
    cache = _TABLES.setdefault(name, None)
    if cache is None or cache.device != device:
        cache = torch.from_numpy(values.copy()).to(device)
        _TABLES[name] = cache
    return cache


def _f16(b: torch.Tensor) -> torch.Tensor:
    """Little-endian fp16 stored in the trailing uint8 pairs of ``b``."""
    return b.contiguous().view(torch.float16).squeeze(-1).float()


def _blocks(raw: torch.Tensor, fmt: QuantFormat, shape: tuple[int, ...]):
    """View raw bytes as (lead, n_blocks, bytes_per_block)."""
    if raw.dtype != torch.uint8:
        raise TypeError(f"dequantize expects uint8 bytes, got {raw.dtype}")
    columns = shape[-1]
    nb = columns // fmt.elements_per_block
    total = fmt.bytes_per_block * nb
    if raw.numel() % total:
        raise ValueError(
            f"{fmt.name}: {raw.numel()} bytes is not whole blocks of {total} for shape {shape}"
        )
    lead = raw.numel() // total
    return raw.view(lead, nb, fmt.bytes_per_block), lead, nb


# --------------------------------------------------------------------------- #
# per-format block decoders: (lead, nb, block_bytes) uint8 -> (lead, nb, 256|32) f32
# --------------------------------------------------------------------------- #


def _deq_f32(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    return blk.contiguous().view(torch.float32)


def _deq_bf16(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    return blk.contiguous().view(torch.bfloat16).float()


def _deq_q8_0(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    d = _f16(blk[:, :, 0:2])
    q = blk[:, :, 2:].contiguous().view(lead, nb, 32).view(torch.int8)
    return (q.float() * d.unsqueeze(-1)).reshape(lead, nb * 32)


def _deq_q2_k(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    scales = blk[:, :, 0:16].to(torch.int32)  # (L,nb,16)
    q = blk[:, :, 16:80].view(lead, nb, 2, 1, 2, 16)  # h, j(unused), s, r
    d = _f16(blk[:, :, 80:82])
    dmin = _f16(blk[:, :, 82:84])
    sc = scales.view(lead, nb, 2, 4, 2)  # h, j, s
    shifts = (2 * torch.arange(4, device=blk.device)).view(1, 1, 1, 4, 1, 1)
    val = (q.expand(lead, nb, 2, 4, 2, 16).to(torch.int32) >> shifts) & 3
    dl = d.view(lead, nb, 1, 1, 1, 1) * (sc & 0xF).unsqueeze(-1)
    ml = dmin.view(lead, nb, 1, 1, 1, 1) * (sc >> 4).unsqueeze(-1)
    return (dl * val - ml).reshape(lead, nb, 256)


def _deq_q4_k(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    d = _f16(blk[:, :, 0:2])
    dmin = _f16(blk[:, :, 2:4])
    scales = blk[:, :, 4:16].to(torch.int32)  # (L,nb,12)
    qs = blk[:, :, 16:144].view(lead, nb, 4, 32).to(torch.int32)  # j64, byte
    # get_scale_min_k4 for j = 0..7
    parts = []
    for j in range(8):
        if j < 4:
            sc = scales[:, :, j] & 63
            m = scales[:, :, j + 4] & 63
        else:
            sc = (scales[:, :, j + 4] & 0xF) | ((scales[:, :, j - 4] >> 6) << 4)
            m = (scales[:, :, j + 4] >> 4) | ((scales[:, :, j] >> 6) << 4)
        parts.append((sc, m))
    sc8 = torch.stack([p[0] for p in parts], dim=-1)  # (L,nb,8)
    m8 = torch.stack([p[1] for p in parts], dim=-1)
    dsc = sc8.view(lead, nb, 4, 2).unsqueeze(-1)
    dm = m8.view(lead, nb, 4, 2).unsqueeze(-1)
    nib = torch.stack([qs & 0xF, qs >> 4], dim=-2)  # (L,nb,4,2,32)
    dl = d.view(lead, nb, 1, 1, 1) * dsc
    ml = dmin.view(lead, nb, 1, 1, 1) * dm
    return (dl * nib - ml).reshape(lead, nb, 256)


def _deq_q6_k(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    ql = blk[:, :, 0:128].view(lead, nb, 2, 64).to(torch.int32)  # h, byte
    qh = blk[:, :, 128:192].view(lead, nb, 2, 32).to(torch.int32)  # h, l
    sc = blk[:, :, 192:208].contiguous().view(torch.int8).to(torch.int32)  # (L,nb,16)
    d = _f16(blk[:, :, 208:210])
    lo = ql[:, :, :, 0:32]  # (L,nb,2,32) == ql[64h + l]
    hi = ql[:, :, :, 32:64]  # ql[64h + 32 + l]
    vals = (
        torch.stack(
            [
                (lo & 0xF) | ((qh & 3) << 4),
                (hi & 0xF) | (((qh >> 2) & 3) << 4),
                (lo >> 4) | (((qh >> 4) & 3) << 4),
                (hi >> 4) | (((qh >> 6) & 3) << 4),
            ],
            dim=-2,  # (L,nb,2,4,32): k, l
        )
        - 32
    )
    # scale index t = is + 2k with is = l//16 -> (k, l) map
    l_idx = torch.arange(32, device=blk.device)
    t_map = (l_idx // 16).view(1, 1, 1, 1, 32) + 2 * torch.arange(4, device=blk.device).view(
        1, 1, 1, 4, 1
    )
    sc_h = sc.view(lead, nb, 2, 1, 8).expand(lead, nb, 2, 4, 8)
    sc_sel = torch.take_along_dim(sc_h, t_map.expand(lead, nb, 2, 4, 32), dim=-1)
    return (d.view(lead, nb, 1, 1, 1) * sc_sel * vals).reshape(lead, nb, 256)


def _deq_iq2_s(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    d = _f16(blk[:, :, 0:2])
    qs = blk[:, :, 2:66]
    idx_b = qs[:, :, 0:32].view(lead, nb, 8, 4).to(torch.int32)  # g, l
    sign_b = qs[:, :, 32:64].view(lead, nb, 8, 4)  # g, l
    qh = blk[:, :, 66:74].view(lead, nb, 8, 1).to(torch.int32)  # g
    scales = blk[:, :, 74:82].view(lead, nb, 8, 1).to(torch.int32)  # g
    l_idx = torch.arange(4, device=blk.device).view(1, 1, 1, 4)
    idx = idx_b | (((qh >> (2 * l_idx)) & 3) << 8)  # (L,nb,8,4)
    grid = _table("iq2s", IQ2S_GRID, blk.device)[idx.long()]  # (L,nb,8,4,8)
    nib = (scales >> (4 * (l_idx // 2))) & 0xF
    db = d.view(lead, nb, 1, 1) * (0.5 + nib) * 0.25
    bits = (
        sign_b.unsqueeze(-1) & _table("kmask", IQ_SIGN_MASK, blk.device).view(1, 1, 1, 1, 8)
    ) != 0
    sign = 1 - 2 * bits.to(torch.int32)
    return (db.unsqueeze(-1) * grid * sign).reshape(lead, nb, 256)


def _deq_iq3_s(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    d = _f16(blk[:, :, 0:2])
    qs = blk[:, :, 2:66].view(lead, nb, 4, 2, 8)  # p, s, byte
    qh = blk[:, :, 66:74].view(lead, nb, 4, 2, 1).to(torch.int32)  # p, s
    signs = blk[:, :, 74:106].view(lead, nb, 4, 2, 4)  # p, s, l
    scales = blk[:, :, 106:110].view(lead, nb, 4).to(torch.int32)  # p
    l_idx = torch.arange(4, device=blk.device).view(1, 1, 1, 1, 4)
    b1 = qs[:, :, :, :, 0:8:2]  # (L,nb,4,2,4) chunk l -> byte 2l
    b2 = qs[:, :, :, :, 1:8:2]
    hi1 = ((qh >> (2 * l_idx)) & 1) << 8
    hi2 = ((qh >> (2 * l_idx + 1)) & 1) << 8
    g1 = _table("iq3s", IQ3S_GRID, blk.device)[(b1.to(torch.int32) | hi1).long()]  # (...,4,4)
    g2 = _table("iq3s", IQ3S_GRID, blk.device)[(b2.to(torch.int32) | hi2).long()]
    grid = torch.stack([g1, g2], dim=-2)  # (L,nb,4,2,4,2,4): p,s,l,g,j
    sc_lo = (scales & 0xF).unsqueeze(-1)
    sc_hi = (scales >> 4).unsqueeze(-1)
    db = d.view(lead, nb, 1, 1) * (1 + 2 * torch.cat([sc_lo, sc_hi], dim=-1))  # p,s
    bits = (
        signs.unsqueeze(-1) & _table("kmask", IQ_SIGN_MASK, blk.device).view(1, 1, 1, 1, 1, 8)
    ) != 0
    sign = (1 - 2 * bits.to(torch.int32)).reshape(lead, nb, 4, 2, 4, 2, 4)  # p,s,l,g,j
    return (db.view(lead, nb, 4, 2, 1, 1, 1) * grid * sign).reshape(lead, nb, 256)


def _deq_iq4_xs(blk: torch.Tensor, lead: int, nb: int) -> torch.Tensor:
    d = _f16(blk[:, :, 0:2])
    scales_h = blk[:, :, 2:4].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF  # (L,nb)
    scales_l = blk[:, :, 4:8].to(torch.int32)  # (L,nb,4)
    qs = blk[:, :, 8:136].view(lead, nb, 8, 16)  # ib, byte
    ib = torch.arange(8, device=blk.device).view(1, 1, 8)
    sel = (
        scales_l[:, :, 0:4]
        .unsqueeze(2)
        .expand(lead, nb, 8, 4)
        .gather(3, (ib // 2).unsqueeze(-1).expand(lead, nb, 8, 1))
    )
    ls = ((sel.squeeze(-1) >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
    dl = d.view(lead, nb, 1) * (ls - 32)  # (L,nb,8)
    nib = torch.stack([qs & 0xF, qs >> 4], dim=-2)  # (L,nb,8,2,16)
    vals = _table("iq4nl", IQ4NL_VALUES, blk.device)[nib.long()]
    return (dl.view(lead, nb, 8, 1, 1) * vals).reshape(lead, nb, 256)


_DECODERS = {
    GGMLType.F32: _deq_f32,
    GGMLType.BF16: _deq_bf16,
    GGMLType.Q8_0: _deq_q8_0,
    GGMLType.Q2_K: _deq_q2_k,
    GGMLType.Q4_K: _deq_q4_k,
    GGMLType.Q6_K: _deq_q6_k,
    GGMLType.IQ2_S: _deq_iq2_s,
    GGMLType.IQ3_S: _deq_iq3_s,
    GGMLType.IQ4_XS: _deq_iq4_xs,
}


def dequantize(raw: torch.Tensor, fmt: QuantFormat, shape: tuple[int, ...]) -> torch.Tensor:
    """Decode ``raw`` uint8 bytes into an f32 tensor of ``shape`` (row-major)."""
    if len(shape) < 1:
        raise ValueError("shape must have at least one dimension")
    decoder = _DECODERS.get(GGMLType(fmt.type_id))
    if decoder is None:
        raise NotImplementedError(f"no decoder for {fmt.name}")
    blk, lead, nb = _blocks(raw, fmt, shape)
    out = decoder(blk, lead, nb)
    return out.reshape(shape)


def dequantize_row(raw: memoryview, fmt: QuantFormat, columns: int) -> torch.Tensor:
    """Decode one row; kept for the phase-1 scaffold contract."""
    buf = torch.from_numpy(__import__("numpy").frombuffer(raw, dtype=__import__("numpy").uint8))
    return dequantize(buf, fmt, (columns,))


__all__ = ["dequantize", "dequantize_row"]
