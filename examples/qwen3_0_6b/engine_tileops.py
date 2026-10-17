"""Engine variant with model components replaced by TileOPs-derived kernels,
running the whole decode step from one captured CUDA graph.

Swapped vs engine.py (TileFoundry-ported kernels):
  * RMSNorm            -> tileops_lite RMSNormKernel          (TileOPs)
  * residual + RMSNorm -> tileops_lite FusedAddRMSNormKernel   (TileOPs)
  * attention          -> tileops_lite gqa_decode_masked       (TileOPs-derived;
                        fixed-capacity KV + device Pos mask, graph-safe)
Kept from the ported TileFoundry kernels: embed, qkv/o/gate_up/down GEMVs
(down keeps its fused silu), qk_rope_cache, lm_head+argmax.

Graph safety: every input to the captured step is a fixed-address buffer, and
the live position lives in a device tensor (read by rope, the masked attention,
and nothing else). TileOPs norm forwards allocate their outputs per call --
during capture those allocations land in the graph's private memory pool and
are replayed at fixed addresses, which is exactly how CUDA graphs handle
intermediate tensors. Sampling is a host argmax over the f32 logits buffer
between replays, like engine.py but without its in-graph sample_step.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E  # noqa: E402
import kernels as K  # noqa: E402


@dataclass(frozen=True)
class Generated:
    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


class EngineTileOps(E.Engine):
    """TileOPs components + one captured CUDA graph for the decode step."""

    def _kernels(self):
        # kept: TileFoundry-ported kernels
        H, HQ, HKV, D, I, V = self.H, self.HQ, self.HKV, self.D, self.I, self.V
        self.k_embed = K.embed(V, H)
        self.k_qkv = K.gemv(H, self.qkv_n, *E.TILES["qkv"])
        self.k_rope = K.qk_rope_cache(
            HQ, HKV, D, self.cap, self.cap, E.TILES["qkv"][2], self.eps
        )
        self.k_o = K.gemv(HQ * D, H, *E.TILES["o"])          # plain GEMV now
        self.k_gu = K.gemv(H, 2 * I, *E.TILES["gate_up"])
        self.k_down = K.gemv_silu(I, H, *E.TILES["down"][:3],
                                  E.TILES["gate_up"][2], E.TILES["down"][3])
        self.k_head, self.nb = K.lm_head(H, V, *E.TILES["head"])
        self.bv = torch.zeros(self.nb, device=self.device, dtype=torch.float32)
        self.bi = torch.zeros(self.nb, device=self.device, dtype=torch.int32)
        self.attn_buf = torch.zeros(self.HQ * self.D, device=self.device,
                                    dtype=torch.bfloat16)

        # swapped in: TileOPs-derived kernels (via tileops_lite copy)
        from tileops_lite.kernels.norm.rms_norm import RMSNormKernel
        from tileops_lite.kernels.norm.fused_add_norm import FusedAddRMSNormKernel
        from tileops_lite.kernels.attention.gqa.masked import gqa_decode_masked
        self.t_norm = RMSNormKernel(H, self.eps, torch.bfloat16)
        self.t_fused = FusedAddRMSNormKernel(H, self.eps, torch.bfloat16)
        self.k_attn_t = gqa_decode_masked(1, HQ, HKV, D, self.cap, self.scale,
                                          "bfloat16")

    def _capture(self):
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step_body()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_body()
        self._reset()

    # ------------------------------------------------------------- one step
    def forward_layer(self, i: int, xn: torch.Tensor, resid: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer *i* given normed input *xn* and residual *resid* (bf16, GPU).
        Position comes from the device ``self.pos`` -- no host value -- so the
        caller can run this inside a captured graph or drive it eagerly."""
        w = self.layers[i]
        kc, vc = self.kc[i], self.vc[i]
        self.k_qkv(xn, w["w_qkv"], self.qkv_part)
        self.k_rope(self.qkv_part, w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                    self.pos, self.pos, kc, vc, self.q)
        self.k_attn_t(self.q.view(1, self.HQ, self.D),
                      kc.view(1, self.cap, self.HKV, self.D),
                      vc.view(1, self.cap, self.HKV, self.D),
                      self.pos,
                      self.attn_buf.view(1, self.HQ, self.D))
        self.k_o(self.attn_buf, w["w_o"], self.o_part)
        o_sum = self.o_part.sum(dim=0).to(torch.bfloat16)
        xn1, h1 = self.t_fused.forward(o_sum, resid, w["gamma_post"])
        self.k_gu(xn1, w["w_gu"], self.gu_part)
        self.k_down(self.gu_part, w["w_down"], self.d_part)
        d_sum = self.d_part.sum(dim=0).to(torch.bfloat16)
        gamma_next = (self.layers[i + 1]["gamma_in"] if i + 1 < self.L
                      else self.gamma_final)
        return self.t_fused.forward(d_sum, h1, gamma_next)

    def _step_body(self) -> None:
        """The captured step: ids/pos come from device tensors, so one capture
        serves every position of a generation."""
        self.k_embed(self.w_embed, self.ids, self.hid)
        resid = self.hid
        xn = self.t_norm.forward(resid, self.layers[0]["gamma_in"])
        for i in range(self.L):
            xn, resid = self.forward_layer(i, xn, resid)
        self.k_head(xn, self.w_head, self.logits, self.bv, self.bi)

    # ------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new: int) -> Generated:
        pl = len(prompt_ids)
        if pl < 1:
            raise ValueError("decode needs a prompt of at least one token")
        if pl + max_new > self.nsteps:
            raise ValueError(
                f"prompt {pl} + {max_new} new exceeds this engine's {self.nsteps} steps"
            )
        self._reset()
        self.kc.zero_()
        self.vc.zero_()
        self.ids.fill_(prompt_ids[0])

        def run_steps(count: int, keep: list[int]) -> None:
            for _ in range(count):
                self.graph.replay()
                nxt = int(self.logits.argmax().item())
                keep.append(nxt)
                pos = int(self.pos.item()) + 1
                self.pos.fill_(pos)
                nxt_token = prompt_ids[pos] if pos < pl else nxt
                self.ids.fill_(nxt_token)

        torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        prefill: list[int] = []
        run_steps(pl - 1, prefill)               # prefill: picks discarded
        torch.cuda.synchronize(self.device)
        t1 = perf_counter()
        out: list[int] = []
        run_steps(max_new, out)                  # the continuation itself
        torch.cuda.synchronize(self.device)
        t2 = perf_counter()
        return Generated(out, t2 - t1, t1 - t0, pl)

    def logits_for(self, prompt_ids: list[int]) -> torch.Tensor:
        self._reset()
        self.kc.zero_()
        self.vc.zero_()
        self.ids.fill_(prompt_ids[0])
        for pos in range(len(prompt_ids)):
            self.graph.replay()
            if pos + 1 < len(prompt_ids):
                self.ids.fill_(prompt_ids[pos + 1])
            self.pos.fill_(pos + 1)
        torch.cuda.synchronize(self.device)
        return self.logits.clone()
