"""Qwen3.6-35B-A3B (qwen35moe) native hybrid decoder.

Placement (docs/qwen36_35b_plan.md section 6): every non-expert tensor is
dequantized once at load and lives on the GPU in bf16; the 256-expert MoE
tensors stay as quant bytes in the GGUF mmap and the selected experts are
staged per token to the compute device where the vectorized dequant + matmul
runs.  A CPU-only build of the same code serves CI (device="cpu").

Numerical conventions (verified against llama.cpp src/models/qwen35moe.cpp):

* GGUF norm gammas are pre-folded (use ``x * gamma`` directly);
* ``ssm_a`` stores ``-exp(A_log)`` so the log-decay is ``a * softplus(..)``;
* the GGUF ``block_count`` includes the MTP block (skipped in the main loop);
* attention ``attn_q`` outputs ``[q | output-gate]`` per head; RoPE rotates
  the leading ``rope_dim`` dims with theta 1e7;
* MoE routing is softmax over all logits, top-k, then renormalize; the
  shared expert is gated by ``sigmoid(x @ ffn_gate_inp_shexp)``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from breezetoken.backends import _torch_gdn_decode
from breezetoken.gguf import GGUFFile
from breezetoken.model.moe import ExpertPool, route
from breezetoken.quant.tensor import QuantizedTensor

#: Prompts at least this long take the batched prefill path.
_PREFILL_BATCH_MIN = 8

#: Prompt handling: prompts are fed token-by-token (batched prefill is phase 6).
_DEFAULT_CONTEXT = 8192


@dataclass(frozen=True)
class Qwen36Config:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    context_length: int
    max_context: int
    n_experts: int
    top_k: int
    expert_ffn: int
    shared_ffn: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_k_dim: int
    gdn_v_dim: int
    gdn_inner: int
    gdn_conv: int
    vocab: int
    eps: float
    l2_eps: float
    ffn: int = 0  # unused, kept for planner-shape compatibility
    nextn_layers: int = 0

    @property
    def attn_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if (i + 1) % self.full_attention_interval == 0)

    @property
    def full_attention_interval(self) -> int:
        return 4

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, max_context: int | None = None) -> Qwen36Config:
        md = gguf.metadata
        context_length = int(md.get("qwen35moe.context_length", 262144))
        requested = _DEFAULT_CONTEXT if max_context is None else int(max_context)
        if requested <= 0:
            raise ValueError("max_context must be positive")
        context = min(context_length, requested)
        return cls(
            hidden=int(md["qwen35moe.embedding_length"]),
            n_layers=int(md["qwen35moe.block_count"])
            - int(md.get("qwen35moe.nextn_predict_layers", 0)),
            n_heads=int(md["qwen35moe.attention.head_count"]),
            n_kv=int(md["qwen35moe.attention.head_count_kv"]),
            head_dim=int(md["qwen35moe.attention.key_length"]),
            rope_dim=int(md.get("qwen35moe.rope.dimension_count", 64)),
            rope_theta=float(md.get("qwen35moe.rope.freq_base", 1.0e7)),
            context_length=context_length,
            max_context=context,
            n_experts=int(md["qwen35moe.expert_count"]),
            top_k=int(md["qwen35moe.expert_used_count"]),
            expert_ffn=int(md["qwen35moe.expert_feed_forward_length"]),
            shared_ffn=int(md["qwen35moe.expert_shared_feed_forward_length"]),
            gdn_key_heads=int(md["qwen35moe.ssm.group_count"]),
            gdn_value_heads=int(md["qwen35moe.ssm.inner_size"])
            // int(md["qwen35moe.ssm.state_size"]),
            gdn_k_dim=int(md["qwen35moe.ssm.state_size"]),
            gdn_v_dim=int(md["qwen35moe.ssm.state_size"]),
            gdn_inner=int(md["qwen35moe.ssm.inner_size"]),
            gdn_conv=int(md["qwen35moe.ssm.conv_kernel"]),
            vocab=int(gguf.tensor_info("token_embd.weight")[0][0]),
            eps=float(md.get("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)),
            l2_eps=1e-6,
            nextn_layers=int(md.get("qwen35moe.nextn_predict_layers", 0)),
        )

    def validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError(f"invalid rope_dim {self.rope_dim}")
        if 2 * self.gdn_key_heads * self.gdn_k_dim != self.gdn_inner:
            raise ValueError("qwen35moe expects 2*key_heads*k_dim == ssm.inner_size")
        if self.gdn_value_heads * self.gdn_v_dim != self.gdn_inner:
            raise ValueError("qwen35moe expects value_heads*v_dim == ssm.inner_size")


@dataclass(frozen=True)
class _HybridPlacement:
    """Component-granular placement summary (non-expert GPU / experts CPU)."""

    max_context: int
    device: str
    n_layers: int

    @property
    def gpu_layers(self) -> int:
        return self.n_layers if self.device.startswith("cuda") else 0

    @property
    def head_on_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def summary(self, total_layers: int) -> str:
        if self.device.startswith("cuda"):
            where = f"hybrid[non-expert cuda / {total_layers} layers experts cpu-quant]"
        else:
            where = "cpu[experts cpu-quant]"
        return f"{where} ctx={self.max_context}"


class _GDNState:
    """Convolution window + recurrent state for one GDN layer."""

    def __init__(self, cfg: Qwen36Config, device: str):
        # conv window covers the qkv projection: 2*key_heads*k_dim (q,k) + inner (v)
        conv_dim = 2 * cfg.gdn_key_heads * cfg.gdn_k_dim + cfg.gdn_inner
        self.conv = torch.zeros(conv_dim, cfg.gdn_conv - 1, device=device)
        inner = cfg.gdn_inner
        self.rec = torch.zeros(
            cfg.gdn_value_heads, cfg.gdn_v_dim, inner // cfg.gdn_value_heads, device=device
        )

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()


class Qwen36Moe:
    """Loaded Qwen3.6-35B-A3B decoder with hybrid quant placement."""

    def __init__(
        self,
        gguf: GGUFFile | str,
        *,
        device: str = "auto",
        dtype: str | torch.dtype | None = None,
        max_context: int | None = None,
        mtp: bool = False,
        **_ignored,
    ) -> None:
        self._use_mtp = mtp
        if isinstance(gguf, (str, __import__("pathlib").Path)):
            gguf = GGUFFile(gguf)
            self._own_gguf = True
        else:
            self._own_gguf = False
        if dtype in ("f32", "float32"):
            self.weight_dtype = torch.float32
        elif dtype in (None, "auto", "bfloat16", "bf16"):
            self.weight_dtype = torch.bfloat16
        else:
            raise ValueError("Qwen36Moe supports dtype auto/bf16/f32")
        self._gguf = gguf
        cfg = Qwen36Config.from_gguf(gguf, max_context)
        cfg.validate()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.placement = _HybridPlacement(cfg.max_context, device, cfg.n_layers)
        self.cfg = cfg
        self.experts = ExpertPool(gguf, cfg.n_layers + max(cfg.nextn_layers, 0), device)
        self.layers: list[dict[str, torch.Tensor]] = []
        self.w: dict[str, torch.Tensor] = {}
        attn = set(cfg.attn_layers)
        self._attn_set = frozenset(attn)
        self._tl = None
        if device.startswith("cuda"):
            try:
                from breezetoken.backends import make_backend

                self._tl = make_backend("tilelang", "cuda", "bfloat16")
            except Exception:
                self._tl = None  # torch fallbacks below keep the model usable
        self._load_nonexpert(gguf)
        self.kv = {
            i: (
                torch.zeros(cfg.n_kv, cfg.max_context, cfg.head_dim, device=device),
                torch.zeros(cfg.n_kv, cfg.max_context, cfg.head_dim, device=device),
            )
            for i in attn
        }
        self.states = {i: _GDNState(cfg, device) for i in range(cfg.n_layers) if i not in attn}
        if self.mtp is not None:
            self.kv[cfg.n_layers] = (
                torch.zeros(cfg.n_kv, cfg.max_context, cfg.head_dim, device=device),
                torch.zeros(cfg.n_kv, cfg.max_context, cfg.head_dim, device=device),
            )
        half = cfg.rope_dim // 2
        inv = cfg.rope_theta ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
        phases = torch.outer(torch.arange(cfg.max_context, device=device, dtype=torch.float32), inv)
        self._rope_cos, self._rope_sin = phases.cos(), phases.sin()

    # ------------------------------------------------------------------ load
    def _weight(self, gguf: GGUFFile, name: str) -> torch.Tensor:
        """2-D weight kept in its native (out, in) GGUF orientation, as bf16.

        Large tensors load in row chunks so the f32 dequant temporary stays
        bounded (head/embedding are 248320 rows).
        """
        qt = QuantizedTensor.from_gguf(gguf, name)
        if len(qt.shape) != 2:
            raise ValueError(f"expected a 2-D weight for {name}, got {qt.shape}")
        chunk = 8192
        if qt.rows <= chunk:
            return qt.load_rows(gguf, 0, qt.rows, self.device).to(self.weight_dtype)
        parts = [
            qt.load_rows(gguf, r, min(chunk, qt.rows - r), self.device).to(self.weight_dtype)
            for r in range(0, qt.rows, chunk)
        ]
        return torch.cat(parts)

    def _norm(self, gguf: GGUFFile, name: str) -> torch.Tensor:
        shape, ttype, _ = gguf.tensor_layout(name)
        if ttype == 0:  # f32
            return gguf.get_tensor(name, dtype=torch.float32).to(self.device)
        qt = QuantizedTensor.from_gguf(gguf, name)
        return qt.load_rows(gguf, 0, qt.rows, self.device).reshape(shape)

    def _load_nonexpert(self, gguf: GGUFFile) -> None:
        cfg = self.cfg
        self.w["embd"] = self._weight(gguf, "token_embd.weight")  # (V, H) bf16
        head = self._weight(gguf, "output.weight")  # (V, H) bf16
        if self._tl is not None:
            # (K, N) layout for the split-K bf16 GEMV kernel.
            self.w["head_kv"] = head.t().contiguous()
            del head
        else:
            self.w["head"] = head
        self.w["norm_f"] = self._norm(gguf, "output_norm.weight")
        for i in range(cfg.n_layers):
            p = f"blk.{i}."
            layer: dict[str, torch.Tensor] = {
                "norm_in": self._norm(gguf, p + "attn_norm.weight"),
                "norm_post": self._norm(gguf, p + "post_attention_norm.weight"),
                "router": self._norm(gguf, p + "ffn_gate_inp.weight"),  # (E, H) f32
                "shexp_gate": self._norm(gguf, p + "ffn_gate_inp_shexp.weight"),
                "shexp_gu": torch.cat(
                    [
                        self._weight(gguf, p + "ffn_gate_shexp.weight"),
                        self._weight(gguf, p + "ffn_up_shexp.weight"),
                    ]
                ),
                "shexp_down": self._weight(gguf, p + "ffn_down_shexp.weight"),
            }
            if i in self._attn_set:
                layer.update(
                    q=self._weight(gguf, p + "attn_q.weight"),
                    k=self._weight(gguf, p + "attn_k.weight"),
                    v=self._weight(gguf, p + "attn_v.weight"),
                    o=self._weight(gguf, p + "attn_output.weight"),
                    q_norm=self._norm(gguf, p + "attn_q_norm.weight"),
                    k_norm=self._norm(gguf, p + "attn_k_norm.weight"),
                )
            else:
                layer.update(
                    qkv=self._weight(gguf, p + "attn_qkv.weight"),
                    z=self._weight(gguf, p + "attn_gate.weight"),
                    out=self._weight(gguf, p + "ssm_out.weight"),
                    conv=self._norm(gguf, p + "ssm_conv1d.weight"),  # (3*inner, conv)
                    w_alpha=self._norm(gguf, p + "ssm_alpha.weight"),  # (nv, H)
                    w_beta=self._norm(gguf, p + "ssm_beta.weight"),
                    a_log=self._norm(gguf, p + "ssm_a"),
                    dt_bias=self._norm(gguf, p + "ssm_dt.bias"),
                    ssm_norm=self._norm(gguf, p + "ssm_norm.weight"),
                )
            self.layers.append(layer)
        self._load_mtp(gguf)

    def _load_mtp(self, gguf: GGUFFile) -> None:
        """MTP/nextn draft block (blk.{n_layers}): attention + MoE + head norms."""
        cfg = self.cfg
        self.mtp: dict[str, torch.Tensor] | None = None
        if cfg.nextn_layers != 1:
            return
        p = f"blk.{cfg.n_layers}."
        try:
            mtp: dict[str, torch.Tensor] = {
                "eh_proj": self._weight(gguf, p + "nextn.eh_proj.weight"),
                "enorm": self._norm(gguf, p + "nextn.enorm.weight"),
                "hnorm": self._norm(gguf, p + "nextn.hnorm.weight"),
                "shnorm": self._norm(gguf, p + "nextn.shared_head_norm.weight"),
            }
        except KeyError:
            return  # checkpoint without the MTP tensors
        layer: dict[str, torch.Tensor] = {
            "norm_in": self._norm(gguf, p + "attn_norm.weight"),
            "norm_post": self._norm(gguf, p + "post_attention_norm.weight"),
            "router": self._norm(gguf, p + "ffn_gate_inp.weight"),
            "shexp_gate": self._norm(gguf, p + "ffn_gate_inp_shexp.weight"),
            "shexp_gu": torch.cat(
                [
                    self._weight(gguf, p + "ffn_gate_shexp.weight"),
                    self._weight(gguf, p + "ffn_up_shexp.weight"),
                ]
            ),
            "shexp_down": self._weight(gguf, p + "ffn_down_shexp.weight"),
            "q": self._weight(gguf, p + "attn_q.weight"),
            "k": self._weight(gguf, p + "attn_k.weight"),
            "v": self._weight(gguf, p + "attn_v.weight"),
            "o": self._weight(gguf, p + "attn_output.weight"),
            "q_norm": self._norm(gguf, p + "attn_q_norm.weight"),
            "k_norm": self._norm(gguf, p + "attn_k_norm.weight"),
        }
        self.layers.append(layer)  # index cfg.n_layers (the MTP block)
        self.mtp = mtp

    # ------------------------------------------------------------------ ops
    @staticmethod
    def _rms(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma

    def _linear(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """w is (out, in); returns f32 (out,)."""
        return (w @ x.reshape(-1).to(w.dtype)).float()

    def _linear_b(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Batched projection: x (S, in) -> f32 (S, out); w is (out, in)."""
        return (x.to(w.dtype) @ w.t()).float()

    def _rope(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.cfg
        rot = x[..., : cfg.rope_dim]
        rest = x[..., cfg.rope_dim :]
        half = cfg.rope_dim // 2
        cos = self._rope_cos[pos].unsqueeze(0)
        sin = self._rope_sin[pos].unsqueeze(0)
        rotated = torch.cat(
            (
                rot[..., :half] * cos - rot[..., half:] * sin,
                rot[..., :half] * sin + rot[..., half:] * cos,
            ),
            dim=-1,
        )
        return torch.cat((rotated, rest), dim=-1)

    def _rope_batch(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        """Batched partial RoPE over positions ``pos0 .. pos0 + S - 1``."""
        cfg = self.cfg
        s = x.shape[0]
        rot = x[..., : cfg.rope_dim]
        rest = x[..., cfg.rope_dim :]
        half = cfg.rope_dim // 2
        cos = self._rope_cos[pos0 : pos0 + s].unsqueeze(1)  # (S, 1, half)
        sin = self._rope_sin[pos0 : pos0 + s].unsqueeze(1)
        rotated = torch.cat(
            (
                rot[..., :half] * cos - rot[..., half:] * sin,
                rot[..., :half] * sin + rot[..., half:] * cos,
            ),
            dim=-1,
        )
        return torch.cat((rotated, rest), dim=-1)

    def _attn_batch(self, i: int, xn: torch.Tensor, pos0: int) -> torch.Tensor:
        """Batched attention over the S new tokens (causal within the batch)."""
        cfg = self.cfg
        layer = self.layers[i]
        H, D = cfg.n_heads, cfg.head_dim
        s = xn.shape[0]
        q_full = self._linear_b(layer["q"], xn).view(s, H, 2 * D)
        q, gate = q_full[..., :D], q_full[..., D:]
        k = self._linear_b(layer["k"], xn).view(s, cfg.n_kv, D)
        v = self._linear_b(layer["v"], xn).view(s, cfg.n_kv, D)
        q = self._rms(q, layer["q_norm"], cfg.eps)
        k = self._rms(k, layer["k_norm"], cfg.eps)
        q = self._rope_batch(q, pos0)
        k = self._rope_batch(k, pos0)
        k_cache, v_cache = self.kv[i]
        k_cache[:, pos0 : pos0 + s] = k.transpose(0, 1)
        v_cache[:, pos0 : pos0 + s] = v.transpose(0, 1)
        seq = pos0 + s
        # Explicit causal band: query row j (absolute position pos0+j) may
        # attend keys 0..pos0+j (is_causal alone would align top-left).
        k_idx = torch.arange(seq, device=xn.device)
        q_idx = pos0 + torch.arange(s, device=xn.device)
        mask = k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)  # (S, seq)
        out = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),  # (1, H, S, D)
                k_cache[:, :seq].unsqueeze(0),
                v_cache[:, :seq].unsqueeze(0),
                attn_mask=mask,
                scale=cfg.attn_scale,
                enable_gqa=True,
            )
            .view(H, s, D)
            .transpose(0, 1)
        )  # (S, H, D)
        out = out * torch.sigmoid(gate)
        return self._linear_b(layer["o"], out.reshape(s, -1))

    def _attn(self, i: int, xn: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        H, D = cfg.n_heads, cfg.head_dim
        q_full = self._linear(layer["q"], xn).view(H, 2 * D)
        q, gate = q_full[:, :D], q_full[:, D:]
        k = self._linear(layer["k"], xn).view(cfg.n_kv, D)
        v = self._linear(layer["v"], xn).view(cfg.n_kv, D)
        q = self._rms(q, layer["q_norm"], cfg.eps)
        k = self._rms(k, layer["k_norm"], cfg.eps)
        q = self._rope(q, pos)
        k = self._rope(k, pos)
        k_cache, v_cache = self.kv[i]
        k_cache[:, pos] = k
        v_cache[:, pos] = v
        seq = pos + 1
        out = F.scaled_dot_product_attention(
            q.view(1, H, 1, D),  # (B, H, L, D)
            k_cache[:, :seq].unsqueeze(0),  # (1, Hkv, S, D)
            v_cache[:, :seq].unsqueeze(0),
            is_causal=False,
            scale=cfg.attn_scale,
            enable_gqa=True,
        ).view(H, D)
        out = out * torch.sigmoid(gate)
        return self._linear(layer["o"], out.reshape(-1))

    def _gdn(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        state = self.states[i]
        hk, hv, dk = cfg.gdn_key_heads, cfg.gdn_value_heads, cfg.gdn_k_dim
        inner = cfg.gdn_inner
        entry = self._linear(layer["qkv"], xn)  # (3*inner,)
        window = torch.cat((state.conv, entry.unsqueeze(1)), dim=1)
        mixed = F.silu((window * layer["conv"]).sum(dim=-1))
        state.conv.copy_(window[:, 1:])
        q = mixed[: inner // 2].view(hk, dk)
        k = mixed[inner // 2 : inner].view(hk, dk)
        v = mixed[inner:].view(hv, dk)
        beta = self._linear(layer["w_beta"], xn)
        g = layer["a_log"].float() * F.softplus(
            self._linear(layer["w_alpha"], xn) + layer["dt_bias"].float()
        )
        if self._tl is not None:
            read = self._tl.gdn_decode(q, k, v, g, beta, state.rec, dk**-0.5, cfg.l2_eps)
        else:
            read = _torch_gdn_decode(q, k, v, g, beta, state.rec, dk**-0.5, cfg.l2_eps)
        z = self._linear(layer["z"], xn).view(hv, dk)
        normed = self._rms(read, layer["ssm_norm"], cfg.eps)
        gated = (normed * F.silu(z)).reshape(-1)
        return self._linear(layer["out"], gated)

    def _gdn_batch(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        """Batched GDN over S tokens; the delta-rule recurrence stays sequential."""
        cfg = self.cfg
        layer = self.layers[i]
        state = self.states[i]
        hk, hv, dk = cfg.gdn_key_heads, cfg.gdn_value_heads, cfg.gdn_k_dim
        inner = cfg.gdn_inner
        s = xn.shape[0]
        entry = self._linear_b(layer["qkv"], xn)  # (S, 3*inner)
        window = torch.cat((state.conv, entry.t()), dim=1)  # (C, 3+S)
        mixed = (
            F.conv1d(window.unsqueeze(0), layer["conv"].unsqueeze(1), groups=window.shape[0])
            .squeeze(0)
            .t()
        )  # (S, C)
        mixed = F.silu(mixed)
        state.conv.copy_(window[:, -layer["conv"].shape[1] + 1 :])
        q = mixed[:, : inner // 2].reshape(s, hk, dk)
        k = mixed[:, inner // 2 : inner].reshape(s, hk, dk)
        v = mixed[:, inner:].reshape(s, hv, dk)
        beta = self._linear_b(layer["w_beta"], xn)  # (S, hv)
        g = layer["a_log"].float() * F.softplus(
            self._linear_b(layer["w_alpha"], xn) + layer["dt_bias"].float()
        )
        reads = torch.empty(s, hv, dk, device=xn.device, dtype=torch.float32)

        if self._tl is not None:

            def decode(q, k, v, g, beta, st, scale, eps):
                return self._tl.gdn_decode(q, k, v, g, beta, st, scale, eps)
        else:

            def decode(q, k, v, g, beta, st, scale, eps):
                return _torch_gdn_decode(q, k, v, g, beta, st, scale, eps)

        for t in range(s):
            reads[t] = decode(
                q[t].contiguous(),
                k[t].contiguous(),
                v[t].contiguous(),
                g[t],
                beta[t],
                state.rec,
                dk**-0.5,
                cfg.l2_eps,
            )
        z = self._linear_b(layer["z"], xn).view(s, hv, dk)
        normed = self._rms(reads, layer["ssm_norm"], cfg.eps)
        gated = (normed * F.silu(z)).reshape(s, -1)
        return self._linear_b(layer["out"], gated)

    def _moe_batch(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        """Batched MoE: one dequant for all selected experts + per-expert matmuls."""
        cfg = self.cfg
        layer = self.layers[i]
        pool = self.experts
        s = xn.shape[0]
        probs = torch.softmax(xn @ layer["router"].t().float(), dim=-1)
        w, ids = torch.topk(probs, cfg.top_k, dim=-1)  # (S, 8)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
        uniq = torch.unique(ids).tolist()
        # one dequant chain for every selected expert (gate|up share a call)
        gate_qt, up_qt, down_qt = (
            pool.layers[i]["gate"],
            pool.layers[i]["up"],
            pool.layers[i]["down"],
        )
        per = gate_qt.rows // gate_qt.shape[0]
        d_per = down_qt.rows // down_qt.shape[0]
        gu = pool._stage_mixed_bf16(
            [(gate_qt, e * per, per) for e in uniq] + [(up_qt, e * per, per) for e in uniq],
            self._gguf,
        )  # (2m*per, H) bf16
        dw = pool._stage_mixed_bf16(
            [(down_qt, e * d_per, d_per) for e in uniq], self._gguf
        )  # (m*d_per, ffn) bf16
        m = len(uniq)
        ffn = cfg.expert_ffn
        hidden = cfg.hidden
        gate_w = gu[: m * per].view(m, per, hidden).to(torch.bfloat16)  # (m, ffn, H)
        up_w = gu[m * per :].view(m, per, hidden).to(torch.bfloat16)
        down_w = dw.view(m, d_per, ffn).to(torch.bfloat16)  # (m, H, ffn)
        pos = torch.empty(cfg.n_experts, dtype=torch.long, device=xn.device)
        pos[torch.tensor(uniq, device=xn.device)] = torch.arange(m, device=xn.device)
        epos = pos[ids]  # (S, 8) slot -> position in uniq
        out = torch.zeros(s, hidden, device=xn.device, dtype=torch.float32)
        xf = xn  # (S, H)
        for j in range(m):
            sel = epos == j  # (S, 8)
            tok = sel.any(dim=-1).nonzero(as_tuple=True)[0]
            if tok.numel() == 0:
                continue
            xe = xf[tok].to(torch.bfloat16)  # (s_j, H)
            w_j = w[tok][sel[tok]].unsqueeze(1)  # (s_j, 1)
            gu_out = xe @ torch.cat([gate_w[j], up_w[j]]).t()  # (s_j, 2*ffn) bf16
            act = F.silu(gu_out[:, :ffn]) * gu_out[:, ffn:]
            out[tok] += w_j * (act @ down_w[j].t()).float()
        ffn_s = cfg.shared_ffn
        gu_s = self._linear_b(layer["shexp_gu"], xn)
        shared = self._linear_b(layer["shexp_down"], F.silu(gu_s[:, :ffn_s]) * gu_s[:, ffn_s:])
        gate = torch.sigmoid(self._linear_b(layer["shexp_gate"], xn)).unsqueeze(-1)
        return out + gate * shared

    def _prefill_batch(self, ids, pos0: int = 0, all_logits: bool = False):
        """Batched forward over S tokens.

        Returns ``(logits, h_last)``: per-position logits ``(S, V)`` when
        ``all_logits`` (speculative verification), else last-token ``(V,)``;
        ``h_last`` is the norm_f hidden of the last token (MTP draft seed).
        """
        cfg = self.cfg
        ids_t = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        h = self.w["embd"][ids_t].float()  # (S, H)
        for i in range(cfg.n_layers):
            layer = self.layers[i]
            xn = self._rms(h, layer["norm_in"], cfg.eps)
            mixer = self._attn_batch(i, xn, pos0) if i in self._attn_set else self._gdn_batch(i, xn)
            h = h + mixer
            xn = self._rms(h, layer["norm_post"], cfg.eps)
            h = h + self._moe_batch(i, xn)
        xn = self._rms(h, self.w["norm_f"], cfg.eps)
        if all_logits:
            if self._tl is not None:
                return self._tl.linear_batch(xn, self.w["head_kv"]).float(), xn[-1]
            return xn @ self.w["head"].float().t(), xn[-1]
        if self._tl is not None:
            return self._tl.linear(xn[-1], self.w["head_kv"]).float(), xn[-1]
        return self._linear(self.w["head"], xn[-1]), xn[-1]

    def _head(self, xn: torch.Tensor) -> torch.Tensor:
        if self._tl is not None:
            return self._tl.linear(xn, self.w["head_kv"]).float()
        return self._linear(self.w["head"], xn)

    def _mtp_forward(self, h_normed: torch.Tensor, token: int, pos: int):
        """One MTP draft step: (trunk hidden, last token) -> (logits, hidden)."""
        cfg = self.cfg
        layer = self.layers[cfg.n_layers]
        mtp = self.mtp
        e = self._rms(self.w["embd"][token].float(), mtp["enorm"], cfg.eps)
        hh = self._rms(h_normed.float(), mtp["hnorm"], cfg.eps)
        cur = self._linear(mtp["eh_proj"], torch.cat([e, hh]))
        xn = self._rms(cur, layer["norm_in"], cfg.eps)
        cur = cur + self._attn(cfg.n_layers, xn, pos)
        xn = self._rms(cur, layer["norm_post"], cfg.eps)
        cur = cur + self._moe(cfg.n_layers, xn)
        hidden = self._rms(cur, mtp["shnorm"], cfg.eps)
        return self._head(hidden), hidden

    def _moe(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        layer = self.layers[i]
        routing = route(xn, layer["router"], self.cfg.top_k)
        routed = self.experts.forward(i, self._gguf, xn, routing)
        gu = self._linear(layer["shexp_gu"], xn)
        ffn = self.cfg.shared_ffn
        shared = self._linear(layer["shexp_down"], F.silu(gu[:ffn]) * gu[ffn:])
        gate = torch.sigmoid(self._linear(layer["shexp_gate"], xn))
        return routed + gate * shared

    # ------------------------------------------------------------------ step
    @torch.inference_mode()
    def forward_token(self, token: int, pos: int, *, logits: bool = True):
        cfg = self.cfg
        if not 0 <= pos < cfg.max_context:
            raise ValueError(f"position {pos} outside [0, {cfg.max_context})")
        h = self.w["embd"][token].float()
        for i in range(cfg.n_layers):
            layer = self.layers[i]
            xn = self._rms(h, layer["norm_in"], cfg.eps)
            mixer = self._attn(i, xn, pos) if i in self._attn_set else self._gdn(i, xn)
            h = h + mixer
            xn = self._rms(h, layer["norm_post"], cfg.eps)
            h = h + self._moe(i, xn)
        if not logits:
            return None
        xn = self._rms(h, self.w["norm_f"], cfg.eps)
        self._last_hnorm = xn
        if self._tl is not None:
            return self._tl.linear(xn, self.w["head_kv"]).float()  # (V,) f32
        return self._linear(self.w["head"], xn)  # (V,) f32

    def _snapshot_states(self):
        return {i: (st.conv.clone(), st.rec.clone()) for i, st in self.states.items()}

    def _restore_states(self, snap) -> None:
        for i, (conv, rec) in snap.items():
            self.states[i].conv.copy_(conv)
            self.states[i].rec.copy_(rec)

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()

    # ------------------------------------------------------------- generate
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> int:
        if temperature <= 0.0:
            return int(logits.argmax())
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative - sorted_probs > top_p
            sorted_probs = torch.where(cutoff, 0.0, sorted_probs)
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def _generate_sampled(
        self,
        ids: Sequence[int],
        max_new: int,
        temperature: float,
        top_p: float,
        eos: frozenset[int],
        cancel: threading.Event | None,
        seed: int | None,
    ) -> Iterator[int]:
        self.reset()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) if seed is not None else 0)
        last = len(ids) - 1
        logits = None
        if len(ids) >= _PREFILL_BATCH_MIN:
            logits, _ = self._prefill_batch(ids)
        else:
            for i, token in enumerate(ids):
                if cancel is not None and cancel.is_set():
                    return
                logits = self.forward_token(token, i, logits=(i == last))
        assert logits is not None
        for step in range(max_new):
            if cancel is not None and cancel.is_set():
                return
            nxt = self._sample(logits, temperature, top_p, generator)
            if nxt in eos:
                return
            yield nxt
            if step + 1 >= max_new:
                return
            if cancel is not None and cancel.is_set():
                return
            logits = self.forward_token(nxt, len(ids) + step, logits=True)

    def _generate_mtp(
        self,
        ids: Sequence[int],
        max_new: int,
        eos: frozenset[int],
        cancel: threading.Event | None,
    ) -> Iterator[int]:
        """Greedy speculative decoding with the MTP/nextn block as the drafter.

        Each iteration drafts ``D`` tokens with the MTP block, verifies them in
        one batched trunk forward, and yields accepted tokens plus one
        correction.  On partial acceptance the GDN states (which the batched
        pass ran past) are restored from a snapshot and the accepted prefix is
        replayed sequentially, so the recurrent state is always exact.
        """
        depth = min(2, self.cfg.max_context - len(ids) - max_new)
        if depth < 1:
            yield from self._generate_plain(ids, max_new, eos, cancel)
            return
        self.reset()
        if len(ids) >= _PREFILL_BATCH_MIN:
            logits, h = self._prefill_batch(ids)
        else:
            logits = None
            for i, t in enumerate(ids):
                logits = self.forward_token(t, i, logits=(i == len(ids) - 1))
            h = self._last_hnorm
        cur_tok = int(ids[-1])
        cur_pos = len(ids) - 1
        produced = 0
        while produced < max_new:
            if cancel is not None and cancel.is_set():
                return
            # draft `depth` tokens from (h, cur_tok)
            drafts: list[int] = []
            h_m, t_m, p_m = h, cur_tok, cur_pos
            for _ in range(depth):
                d_logits, h_m = self._mtp_forward(h_m, t_m, p_m)
                d_tok = int(d_logits.argmax())
                drafts.append(d_tok)
                t_m, p_m = d_tok, p_m + 1
            # verify the drafts in one batched trunk pass
            snap = self._snapshot_states()
            v_logits, v_h = self._prefill_batch(drafts, pos0=cur_pos + 1, all_logits=True)
            preds = [logits] + [v_logits[j] for j in range(depth - 1)]
            k = 0
            while k < depth and int(preds[k].argmax()) == drafts[k]:
                k += 1
            if k == depth:
                for d_tok in drafts:
                    if produced >= max_new:
                        return
                    if cancel is not None and cancel.is_set():
                        return
                    if d_tok in eos:
                        return
                    yield d_tok
                    produced += 1
                cur_tok, cur_pos = drafts[-1], cur_pos + depth
                logits, h = v_logits[-1], v_h
                continue
            # partial acceptance: rewind and replay the accepted prefix plus the
            # correction so the recurrent state is exact and the correction is
            # never yielded twice.
            self._restore_states(snap)
            for j in range(k):
                self.forward_token(drafts[j], cur_pos + 1 + j, logits=False)
            accepted = drafts[:k]
            for d_tok in accepted:
                if produced >= max_new:
                    return
                if d_tok in eos:
                    return
                yield d_tok
                produced += 1
            correction = int(preds[k].argmax())
            if produced >= max_new:
                return
            if correction in eos:
                return
            yield correction
            produced += 1
            logits = self.forward_token(correction, cur_pos + k + 1, logits=True)
            h = self._last_hnorm
            cur_tok, cur_pos = correction, cur_pos + k + 1

    def _generate_plain(
        self,
        ids: Sequence[int],
        max_new: int,
        eos: frozenset[int],
        cancel: threading.Event | None,
    ) -> Iterator[int]:
        self.reset()
        last = len(ids) - 1
        logits = None
        if len(ids) >= _PREFILL_BATCH_MIN:
            logits, _ = self._prefill_batch(ids)
        else:
            for i, token in enumerate(ids):
                if cancel is not None and cancel.is_set():
                    return
                logits = self.forward_token(token, i, logits=(i == last))
        assert logits is not None
        for step in range(max_new):
            if cancel is not None and cancel.is_set():
                return
            nxt = self._sample(logits, 0.0, 1.0, None)
            if nxt in eos:
                return
            yield nxt
            if step + 1 >= max_new:
                return
            if cancel is not None and cancel.is_set():
                return
            logits = self.forward_token(nxt, len(ids) + step, logits=True)

    def generate_stream(
        self,
        ids: Sequence[int],
        max_new: int,
        temperature: float = 0.0,
        *,
        top_p: float = 1.0,
        eos_ids: Sequence[int] = (),
        cancel: threading.Event | None = None,
        seed: int | None = None,
    ) -> Iterator[int]:
        if not ids:
            raise ValueError("prompt must contain at least one token")
        if max_new < 0:
            raise ValueError("max_new must be >= 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if len(ids) + max_new > self.cfg.max_context:
            raise ValueError(
                f"prompt {len(ids)} + max_new {max_new} exceeds max_context {self.cfg.max_context}"
            )
        if max_new == 0:
            return
        if cancel is not None and cancel.is_set():
            return

        eos = frozenset(eos_ids)
        if (
            self._use_mtp
            and self.mtp is not None
            and temperature <= 0
            and self.device.startswith("cuda")
        ):
            yield from self._generate_mtp(ids, max_new, eos, cancel)
            return
        if temperature > 0:
            yield from self._generate_sampled(ids, max_new, temperature, top_p, eos, cancel, seed)
            return
        yield from self._generate_plain(ids, max_new, eos, cancel)

    def generate(
        self,
        ids: Sequence[int],
        max_new: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[int]:
        return list(self.generate_stream(ids, max_new, temperature, **kwargs))

    def warmup(self) -> None:
        self.reset()
        self.forward_token(0, 0, logits=True)
        self.reset()

    def close(self) -> None:
        self.layers.clear()
        self.w.clear()
        self.kv.clear()
        self.states.clear()
        if self._tl is not None:
            self._tl.close()
            self._tl = None
        if self._own_gguf:
            self._gguf.close()

    @property
    def backend(self):
        class _Info:
            name = "hybrid-quant"

        return _Info()

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"Qwen36Moe(hidden={cfg.hidden}, layers={cfg.n_layers}, "
            f"experts={cfg.n_experts}top{cfg.top_k}, max_context={cfg.max_context}, "
            f"{self.placement.summary(cfg.n_layers)}, dtype={self.weight_dtype})"
        )


__all__ = ["Qwen36Config", "Qwen36Moe"]
