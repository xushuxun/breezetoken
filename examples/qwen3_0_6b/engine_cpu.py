"""CPU inference engine: every model component is a tilelang kernel compiled
for the CPU target ("c" backend, cc flags in kernels_cpu.CC_FLAGS).

float32 end to end (the C codegen cannot emit bfloat16). Single-threaded
scalar-with-SIMD kernels: expect single-digit tok/s on a 0.6B -- this engine
is the correctness reference and the CPU half of engine_hybrid.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E  # noqa: E402  (for weight loading, config, rope caches)


@dataclass(frozen=True)
class Generated:
    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


class EngineCPU(E.Engine):
    """Engine contract, CPU device, tilelang CPU kernels for all components."""

    def __init__(self, ckpt, **kw):
        kw.setdefault("device", "cpu")
        self.dt = torch.float32
        super().__init__(ckpt, **kw)
        self.cos = self.cos.float().contiguous()
        self.sin = self.sin.float().contiguous()

    # the parent's __init__ reads self.dt through _load; make sure it is f32
    # even though Engine.__init__ sets bf16 after our early assignment
    def _load(self, ckpt):
        self.dt = torch.float32
        super()._load(ckpt)

    def _capture(self):
        self._reset()

    def graph(self):
        raise RuntimeError("EngineCPU runs eagerly; no captured graph")

    # ---------------------------------------------------------------- kernels
    def _kernels(self):
        import kernels_cpu as KC
        H, HQ, HKV, D, I, V = self.H, self.HQ, self.HKV, self.D, self.I, self.V
        self.k_embed = KC.embed(V, H)
        self.k_norm = KC.rms_norm(H, self.eps)
        self.k_fused = KC.resid_rms_norm(H, self.eps)
        self.k_qkv = KC.gemv(H, self.qkv_n)
        self.k_rope = KC.qk_rope_cache(HQ, HKV, D, self.cap, self.eps)
        self.k_attn = KC.attn_decode(HQ, HKV, D, self.cap, self.scale)
        self.k_o = KC.gemv(HQ * D, H)
        self.k_gu = KC.gemv(H, 2 * I)
        self.k_silu = KC.silu_mul(I)
        self.k_down = KC.gemv(I, H)
        self.k_head = KC.gemv(H, V)

    # ---------------------------------------------------------------- buffers
    def _buffers(self):
        dev, dt = self.device, torch.float32

        def z(*shape):
            return torch.zeros(*shape, device=dev, dtype=dt)

        self.hid = z(self.H)
        self.xn = z(self.H)
        self.h1 = z(self.H)
        self.xn1 = z(self.H)
        self.attn_out = z(self.HQ * self.D)
        self.gu = z(2 * self.I)
        self.act = z(self.I)
        self.dout = z(self.H)
        self.q = z(self.HQ * self.D)
        self.qkv_buf = z(self.qkv_n)
        self.obuf = z(self.H)
        self.logits = z(self.V)
        self.kc = z(self.L, self.cap, self.HKV * self.D)
        self.vc = z(self.L, self.cap, self.HKV * self.D)
        self.pos = torch.zeros(1, dtype=torch.int32)
        self.ids = torch.zeros(1, dtype=torch.int64)

    def _reset(self):
        self.pos.zero_()
        self.kc.zero_()
        self.vc.zero_()

    # ------------------------------------------------------------------- step
    def forward_layer(self, i: int, xn: torch.Tensor, resid: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer *i* given normed input *xn* and residual *resid* (f32 on CPU);
        returns (xn_next, resid_next)."""
        w = self.layers[i]
        kc, vc = self.kc[i], self.vc[i]
        self.k_qkv(xn, w["w_qkv"], self.qkv_buf)
        self.k_rope(self.qkv_buf, w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                    self.pos, kc, vc, self.q)
        self.k_attn(self.q, kc, vc, self.pos, self.attn_out)
        self.k_o(self.attn_out, w["w_o"], self.obuf)
        self.k_fused(resid, self.obuf, w["gamma_post"], self.h1, self.xn1)
        self.k_gu(self.xn1, w["w_gu"], self.gu)
        self.k_silu(self.gu, self.act)
        self.k_down(self.act, w["w_down"], self.dout)
        gamma_next = (self.layers[i + 1]["gamma_in"] if i + 1 < self.L
                      else self.gamma_final)
        self.k_fused(self.h1, self.dout, gamma_next, self.hid, self.xn)
        return self.xn, self.hid

    def _step_eager(self) -> None:
        self.k_embed(self.w_embed, self.ids, self.hid)
        self.k_norm(self.hid, self.layers[0]["gamma_in"], self.xn)
        for i in range(self.L):
            self.xn, self.hid = self.forward_layer(i, self.xn, self.hid)
        self.k_head(self.xn, self.w_head, self.logits)

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
                self._step_eager()
                nxt = int(self.logits.argmax().item())
                keep.append(nxt)
                s = int(self.pos.item())
                nxt_token = prompt_ids[s + 1] if s + 1 < pl else nxt
                self.ids.fill_(nxt_token)
                self.pos.fill_(s + 1)

        t0 = perf_counter()
        prefill: list[int] = []
        run_steps(pl - 1, prefill)
        t1 = perf_counter()
        out: list[int] = []
        run_steps(max_new, out)
        t2 = perf_counter()
        return Generated(out, t2 - t1, t1 - t0, pl)

    def logits_for(self, prompt_ids: list[int]) -> torch.Tensor:
        self._reset()
        self.kc.zero_()
        self.vc.zero_()
        self.ids.fill_(prompt_ids[0])
        for s in range(len(prompt_ids)):
            self._step_eager()
            if s + 1 < len(prompt_ids):
                self.ids.fill_(prompt_ids[s + 1])
            self.pos.fill_(s + 1)
        return self.logits.clone()
