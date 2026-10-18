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

    def _stage_mixed(self, parts, gguf: GGUFFile) -> torch.Tensor:
        """Stage ``[(qt, row, count)]`` byte ranges as one (rows, columns) f32 tile."""
        fmt0 = parts[0][0].format
        rows = sum(cnt for _, _, cnt in parts)
        for qt, _, _ in parts:
            if qt.format.type_id != fmt0.type_id or qt.columns != parts[0][0].columns:
                raise ValueError("_stage_mixed requires identical formats/layouts")
        device = self.device
        if device.startswith("cuda"):
            nbytes = sum(cnt * qt.row_bytes for qt, _, cnt in parts)
            pin = self._pin(nbytes)
            view = pin.numpy()
            pos = 0
            for qt, row, cnt in parts:
                n = cnt * qt.row_bytes
                np.copyto(
                    view[pos : pos + n],
                    gguf.bytes_view(qt.offset + row * qt.row_bytes, n),
                )
                pos += n
            raw = pin[:pos].to(device, non_blocking=True)
        else:
            chunks = [np.ascontiguousarray(qt.bytes_view(gguf, row, cnt)) for qt, row, cnt in parts]
            raw = torch.from_numpy(np.concatenate(chunks))
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

    # ---------------------------------------------------------------- forward
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
