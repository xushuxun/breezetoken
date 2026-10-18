"""MoE routing and quantized expert execution for the native Qwen3.6 path.

The router follows the reference (llama.cpp ``build_moe_ffn`` with SOFTMAX
gating): softmax over all expert logits, top-k, then renormalize the selected
weights.  Expert weights stay quantized bytes in the GGUF mmap; per decode
step the selected experts' rows are staged to the compute device and the
dequant+matmul runs there (``kernels/*/quant_gemv`` semantics) -- no full
expert tensor is ever materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from breezetoken.gguf import GGUFFile
from breezetoken.quant.dequant import dequantize
from breezetoken.quant.tensor import QuantizedTensor

#: clamp floor from llama.cpp build_moe_ffn (smallest f16) to avoid div-by-zero
_WEIGHT_SUM_MIN = 6.103515625e-5


@dataclass(frozen=True)
class RouterOutput:
    weights: torch.Tensor  # (top_k,) f32, renormalized
    expert_ids: torch.Tensor  # (top_k,) int64


def route(x: torch.Tensor, gate_inp: torch.Tensor, top_k: int) -> RouterOutput:
    """Softmax top-k routing; ``gate_inp`` is (n_experts, hidden) f32."""
    logits = gate_inp @ x.reshape(-1).float()
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, top_k)
    weights = weights / weights.sum().clamp_min(_WEIGHT_SUM_MIN)
    return RouterOutput(weights=weights, expert_ids=ids.to(torch.int64))


class ExpertPool:
    """Per-layer expert tensors as quant descriptors + fused decode execution."""

    def __init__(self, gguf: GGUFFile, n_layers: int, device: str) -> None:
        self.device = device
        self.layers: list[dict[str, QuantizedTensor]] = []
        for i in range(n_layers):
            self.layers.append(
                {
                    "gate": QuantizedTensor.from_gguf(gguf, f"blk.{i}.ffn_gate_exps.weight"),
                    "up": QuantizedTensor.from_gguf(gguf, f"blk.{i}.ffn_up_exps.weight"),
                    "down": QuantizedTensor.from_gguf(gguf, f"blk.{i}.ffn_down_exps.weight"),
                }
            )
        self._pins: dict[tuple[str, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ stage
    def _pin(self, nbytes: int) -> torch.Tensor:
        buf = self._pins.get((self.device, nbytes))
        if buf is None:
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.device.startswith("cuda"))
            self._pins[(self.device, nbytes)] = buf
        return buf

    def _stage_bytes_mixed(self, parts, gguf: GGUFFile) -> torch.Tensor:
        """Stage ``[(qt, row, count)]`` byte ranges as raw quant bytes on-device."""
        nbytes = sum(cnt * qt.row_bytes for qt, _, cnt in parts)
        if self.device.startswith("cuda"):
            pin = self._pin(nbytes)
            pos = 0
            for qt, row, cnt in parts:
                n = cnt * qt.row_bytes
                # torch copy_ uses the OpenMP-parallel CPU path, several times
                # faster than single-threaded np.copyto for these ~0.5 MB slices.
                pin[pos : pos + n].copy_(
                    torch.from_numpy(gguf.bytes_view(qt.offset + row * qt.row_bytes, n))
                )
                pos += n
            return pin[:pos].to(self.device, non_blocking=True)
        chunks = [np.ascontiguousarray(qt.bytes_view(gguf, row, cnt)) for qt, row, cnt in parts]
        return torch.from_numpy(np.concatenate(chunks))

    def _stage_mixed_bf16(self, parts, gguf: GGUFFile) -> torch.Tensor:
        """Like :meth:`_stage_mixed` but dequantizes to bf16 via the CUDA kernel.

        Rows are zero-padded to a bucket multiple so the per-(rows, columns)
        kernel JIT compiles a handful of shapes instead of one per layer.
        """
        from breezetoken.kernels.cuda.quant_gemv import dequant_tile

        fmt0 = parts[0][0].format
        rows = sum(cnt for _, _, cnt in parts)
        for qt, _, _ in parts:
            if qt.format.type_id != fmt0.type_id or qt.columns != parts[0][0].columns:
                raise ValueError("_stage_mixed_bf16 requires identical formats/layouts")
        columns = parts[0][0].columns
        row_bytes = fmt0.row_bytes(columns)
        bucket = 8192
        rows_p = max(bucket, -(-rows // bucket) * bucket)
        nbytes = rows * row_bytes
        if self.device.startswith("cuda"):
            pin = self._pin(rows_p * row_bytes)
            pos = 0
            for qt, row, cnt in parts:
                n = cnt * qt.row_bytes
                pin[pos : pos + n].copy_(
                    torch.from_numpy(gguf.bytes_view(qt.offset + row * qt.row_bytes, n))
                )
                pos += n
            pin[pos:].zero_()  # keep the padding decode-safe (d=0 -> zeros)
            raw = pin[: rows_p * row_bytes].to(self.device, non_blocking=True)
        else:
            chunks = [np.ascontiguousarray(qt.bytes_view(gguf, row, cnt)) for qt, row, cnt in parts]
            pad = rows_p * row_bytes - nbytes
            if pad:
                chunks.append(np.zeros(pad, dtype=np.uint8))
            raw = torch.from_numpy(np.concatenate(chunks))
        out = dequant_tile(raw, fmt0, rows_p, columns)
        return out[:rows]

    def _stage_mixed(self, parts, gguf: GGUFFile) -> torch.Tensor:
        """Stage ``[(qt, row, count)]`` byte ranges as one (rows, columns) f32 tile."""
        fmt0 = parts[0][0].format
        rows = sum(cnt for _, _, cnt in parts)
        for qt, _, _ in parts:
            if qt.format.type_id != fmt0.type_id or qt.columns != parts[0][0].columns:
                raise ValueError("_stage_mixed requires identical formats/layouts")
        raw = self._stage_bytes_mixed(parts, gguf)
        return dequantize(raw, fmt0, (rows, parts[0][0].columns))

    def _stage_experts(
        self, qt: QuantizedTensor, gguf: GGUFFile, experts: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize ``len(experts)`` expert slices stacked as (n, out, in) f32."""
        per = qt.rows // qt.shape[0]  # rows per expert
        parts = [(qt, int(e) * per, per) for e in experts.tolist()]
        return self._stage_mixed(parts, gguf).view(len(parts), per, qt.columns)

    def _stage_gate_up(
        self, gate: QuantizedTensor, up: QuantizedTensor, gguf: GGUFFile, experts: torch.Tensor
    ) -> torch.Tensor:
        """gate and up share shape/format: one stage + one fused GEMV worth of rows."""
        per = gate.rows // gate.shape[0]
        ids = [int(e) for e in experts.tolist()]
        parts = [(gate, e * per, per) for e in ids] + [(up, e * per, per) for e in ids]
        return self._stage_mixed(parts, gguf)  # (2n*per, columns)

    def stage_expert(
        self, layer: dict[str, QuantizedTensor], gguf: GGUFFile, expert: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantized (gate, up, down) f32 matrices for one expert."""
        ids = torch.tensor([expert])
        return (
            self._stage_experts(layer["gate"], gguf, ids)[0],
            self._stage_experts(layer["up"], gguf, ids)[0],
            self._stage_experts(layer["down"], gguf, ids)[0],
        )

    # ---------------------------------------------------------------- forward
    def _forward_fused(
        self,
        layer: dict[str, QuantizedTensor],
        gguf: GGUFFile,
        xin: torch.Tensor,
        ids: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """CUDA path: fused dequant+GEMV kernels straight from staged bytes."""
        from breezetoken.kernels.cuda.quant_gemv import (
            iq2s_gemv,
            iq2s_table,
            iq4nl_table,
            iq4xs_gemv,
        )

        device = self.device
        ffn = layer["gate"].shape[1]
        per = layer["gate"].rows // layer["gate"].shape[0]
        ids_list = [int(e) for e in ids.tolist()]
        gu_parts = [(layer["gate"], e * per, per) for e in ids_list] + [
            (layer["up"], e * per, per) for e in ids_list
        ]
        d_per = layer["down"].rows // layer["down"].shape[0]
        d_parts = [(layer["down"], e * d_per, d_per) for e in ids_list]
        # one pinned buffer + one H2D for gate/up and down together
        raw = self._stage_bytes_mixed(gu_parts + d_parts, gguf)
        gu_nbytes = sum(cnt * qt.row_bytes for qt, _, cnt in gu_parts)
        gu_raw = raw[:gu_nbytes]
        down_raw = raw[gu_nbytes:]
        gu = iq2s_gemv(xin, gu_raw, iq2s_table(device), 2 * n * per, layer["gate"].columns)
        g = gu[: n * ffn].view(n, ffn)
        u = gu[n * ffn :].view(n, ffn)
        act = F.silu(g) * u  # (n, ffn)
        out = iq4xs_gemv(
            act.reshape(-1),
            down_raw,
            iq4nl_table(device),
            n * d_per,
            layer["down"].columns,
            rows_per_expert=d_per,
        )
        return out.view(n, -1)

    def forward(
        self,
        i: int,
        gguf: GGUFFile,
        x: torch.Tensor,
        routing: RouterOutput,
    ) -> torch.Tensor:
        """Routed expert output for one token: sum_k w_k * expert_k(x)."""
        ids = routing.expert_ids
        w = routing.weights
        layer = self.layers[i]
        xin = x.reshape(-1).float()
        n = ids.numel()
        if (
            self.device.startswith("cuda")
            and layer["gate"].format.name == "iq2_s"
            and layer["down"].format.name == "iq4_xs"
        ):
            out = self._forward_fused(layer, gguf, xin, ids, n)
            return (out * w.view(-1, 1).to(out.device)).sum(dim=0)
        ffn = layer["gate"].shape[1]
        gu = self._stage_gate_up(layer["gate"], layer["up"], gguf, ids)
        gu = gu @ xin  # (2n*ffn,) = [gate_e for e in ids] ++ [up_e for e in ids]
        g = gu[: n * ffn].view(n, ffn)
        u = gu[n * ffn :].view(n, ffn)
        act = F.silu(g) * u  # (n, ffn)
        down = self._stage_experts(layer["down"], gguf, ids)
        out = down @ act.unsqueeze(-1)  # (n, hidden, 1)
        return (out.squeeze(-1) * w.view(-1, 1).to(out.device)).sum(dim=0)


__all__ = ["ExpertPool", "RouterOutput", "route"]
