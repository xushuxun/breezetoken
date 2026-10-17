"""CPU-GPU hybrid inference: per-layer device placement with the residual
stream crossing device boundaries as a 4 KB f32 vector.

Composition, not duplication: owns one EngineTileOps (GPU half) and one
EngineCPU (CPU half) and drives their per-layer ``forward_layer`` methods.
Device maps:

  all-gpu     every module on GPU        (sanity: ~= EngineTileOps)
  all-cpu     every module on CPU        (sanity: == EngineCPU)
  alternating even blocks on GPU, odd on CPU (maximal real cross-device traffic)

The residual stream master copy is f32 on the host. A GPU block receives
(xn, resid) as bf16 and returns fresh bf16 tensors that are converted back;
per boundary that is 2 H2D + 2 D2H transfers of 2 KB plus one sync.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_cpu as EC  # noqa: E402
import engine_tileops as ET  # noqa: E402


@dataclass(frozen=True)
class Generated:
    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


class EngineHybrid:
    """Layer-granularity hybrid of the TileOPs-GPU and tilelang-CPU engines."""

    def __init__(self, ckpt, device_map: str = "alternating", *,
                 max_new: int = 128, prompt_room: int = 64):
        self.gpu = ET.EngineTileOps(ckpt, device="cuda:0",
                                    max_new=max_new, prompt_room=prompt_room)
        self.cpu = EC.EngineCPU(ckpt, max_new=max_new, prompt_room=prompt_room)
        self.L = self.cpu.L
        self.device_map = device_map
        self.map = self._build_map(device_map)
        self.nsteps = self.cpu.nsteps
        self.device = "cpu"  # residual master lives here

    def _build_map(self, policy: str) -> dict:
        if policy == "all-gpu":
            blocks = ["gpu"] * self.L
        elif policy == "all-cpu":
            blocks = ["cpu"] * self.L
        elif policy == "alternating":
            blocks = ["gpu" if i % 2 == 0 else "cpu" for i in range(self.L)]
        else:
            raise ValueError(f"unknown device map: {policy}")
        return {"embed": "gpu", "head": "gpu", "blocks": blocks}

    # ------------------------------------------------------------------- step
    def _step(self, token: int, pos: int) -> None:
        """Advance one position; leaves next-token logits in self._logits."""
        g, c = self.gpu, self.cpu
        g.pos.fill_(pos)
        c.pos.fill_(pos)

        # embed (always fast; keep it on the GPU map entry for realism)
        g.ids.fill_(token)
        g.k_embed(g.w_embed, g.ids, g.hid)
        resid = g.hid.float().cpu()                      # f32 master copy
        c.k_norm(resid, c.layers[0]["gamma_in"], c.xn)   # input norm on CPU
        xn = c.xn.clone()

        for i in range(self.L):
            if self.map["blocks"][i] == "gpu":
                xn_g = xn.to("cuda", torch.bfloat16)
                resid_g = resid.to("cuda", torch.bfloat16)
                xn_g, resid_g = g.forward_layer(i, xn_g, resid_g)
                xn = xn_g.float().cpu()
                resid = resid_g.float().cpu()
            else:
                xn, resid = c.forward_layer(i, xn, resid)

        if self.map["head"] == "gpu":
            xn_g = xn.to("cuda", torch.bfloat16)
            g.k_head(xn_g, g.w_head, g.logits, g.bv, g.bi)
            self._logits = g.logits.clone().cpu()
        else:
            c.k_head(xn, c.w_head, c.logits)
            self._logits = c.logits.clone()

    # ------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new: int) -> Generated:
        pl = len(prompt_ids)
        if pl + max_new > self.nsteps:
            raise ValueError(
                f"prompt {pl} + {max_new} new exceeds this engine's {self.nsteps} steps"
            )
        self.gpu._reset()
        self.gpu.kc.zero_()
        self.gpu.vc.zero_()
        self.cpu._reset()
        self.cpu.kc.zero_()
        self.cpu.vc.zero_()

        def run_steps(count: int, keep: list[int]) -> None:
            for s in range(count):
                pos = self._pos
                self._step(self._cur, pos)
                nxt = int(self._logits.argmax().item())
                keep.append(nxt)
                self._cur = prompt_ids[pos + 1] if pos + 1 < pl else nxt
                self._pos = pos + 1

        self._pos = 0
        self._cur = prompt_ids[0]
        t0 = perf_counter()
        prefill: list[int] = []
        run_steps(pl - 1, prefill)
        t1 = perf_counter()
        out: list[int] = []
        run_steps(max_new, out)
        t2 = perf_counter()
        return Generated(out, t2 - t1, t1 - t0, pl)

    def logits_for(self, prompt_ids: list[int]) -> torch.Tensor:
        self.gpu._reset()
        self.gpu.kc.zero_()
        self.gpu.vc.zero_()
        self.cpu._reset()
        self.cpu.kc.zero_()
        self.cpu.vc.zero_()
        self._pos = 0
        self._cur = prompt_ids[0]
        for pos in range(len(prompt_ids)):
            self._step(self._cur, pos)
            if pos + 1 < len(prompt_ids):
                self._cur = prompt_ids[pos + 1]
            self._pos = pos + 1
        return self._logits.clone()

    def describe(self) -> str:
        rows = [f"embed -> {self.map['embed']}", f"head  -> {self.map['head']}"]
        blocks = self.map["blocks"]
        # compress runs
        runs, start = [], 0
        for i in range(1, len(blocks) + 1):
            if i == len(blocks) or blocks[i] != blocks[start]:
                rows.append(f"block.{start}-{i-1} -> {blocks[start]}")
                start = i
        return "\n".join(rows)
