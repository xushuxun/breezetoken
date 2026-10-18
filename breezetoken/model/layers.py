"""The three native decoder operations: attention, GDN and the dense MLP.

They are mixins rather than free functions so the hot path keeps the model's
loaded weights, layer backends and caches close at hand.  ``qwen35.py`` owns
loading and generation; this file owns per-layer arithmetic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from breezetoken.backends import Backend
from breezetoken.model.config import GDNState


class LayerMixin:
    # ----------------------------------------------------------------- pieces
    def _embed(self, token: int) -> torch.Tensor:
        if token < 0 or token >= self.cfg.vocab:
            raise ValueError(f"token {token} out of range [0, {self.cfg.vocab})")
        # head is (H, V): a column is the token embedding.  The copy is 4 KB
        # and keeps the strided read out of the hot residual path.
        return self.w["head"][:, token].float().contiguous()

    def _rope_tables(self, cap: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        half = cfg.rope_dim // 2
        inv = cfg.rope_theta ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
        phases = torch.outer(torch.arange(cap, device=device, dtype=torch.float32), inv)
        freqs = torch.cat((phases, phases), dim=-1)
        return freqs.cos(), freqs.sin()

    def _ensure_rope_state(self, device: str) -> None:
        if device in self._pos_bufs:
            return
        self._rope_tables_by_device[device] = self._rope_tables(self.cfg.max_context, device)
        self._pos_bufs[device] = torch.zeros(1, dtype=torch.int32, device=device)

    def _rope(self, x: torch.Tensor, backend: Backend) -> torch.Tensor:
        device = backend.device
        self._ensure_rope_state(device)
        dim = x.shape[-1]
        rows = x.numel() // dim
        cos, sin = self._rope_tables_by_device[device]
        out = backend.rope(
            x.reshape(rows, dim).contiguous(),
            cos,
            sin,
            self._pos_bufs[device],
            self.cfg.rope_dim,
        )
        return out.reshape(x.shape)

    def _attn(self, i: int, xn: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        backend = self.layer_backends[i]

        qkv = backend.linear(xn, layer["w_qkv_attn"])
        q_size = cfg.n_heads * 2 * cfg.head_dim
        kv_size = cfg.n_kv * cfg.head_dim
        qg = qkv[:q_size].view(1, cfg.n_heads, 2 * cfg.head_dim)
        q, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
        k = qkv[q_size : q_size + kv_size].view(1, cfg.n_kv, cfg.head_dim)
        v = qkv[q_size + kv_size :].view(1, cfg.n_kv, cfg.head_dim)

        q = self._rope(backend.rms_norm(q, layer["q_norm"], cfg.eps), backend)
        k = self._rope(backend.rms_norm(k, layer["k_norm"], cfg.eps), backend)

        cache = self.kv[i]
        cache.k[pos] = k.reshape(-1)
        cache.v[pos] = v.reshape(-1)
        seq = pos + 1
        out = backend.attention(
            q.view(cfg.n_heads, cfg.head_dim),
            cache.k[:seq].view(seq, cfg.n_kv, cfg.head_dim),
            cache.v[:seq].view(seq, cfg.n_kv, cfg.head_dim),
            cfg.attn_scale,
        )
        out = out.view(1, cfg.n_heads, cfg.head_dim) * torch.sigmoid(gate.float())
        return backend.linear(out.reshape(1, -1), layer["w_o"])

    def _gdn(self, i: int, xn: torch.Tensor, state: GDNState) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        backend = self.layer_backends[i]
        hv, dk, dv = cfg.ssm_heads, cfg.ssm_state, cfg.ssm_state

        entry = backend.linear(xn, layer["w_qkv"])
        window = torch.cat((state.conv, entry.reshape(-1, 1)), dim=1)
        mixed = F.silu((window * layer["conv"]).sum(dim=-1))
        state.conv.copy_(window[:, 1:])

        q = mixed[: cfg.ssm_inner].view(hv, dk)
        k = mixed[cfg.ssm_inner : 2 * cfg.ssm_inner].view(hv, dk)
        v = mixed[2 * cfg.ssm_inner :].view(hv, dv)

        beta_logits = backend.linear(xn, layer["w_beta"]).view(hv)
        # GGUF ssm_a is already -exp(A_log); do not exp() it again.
        g = layer["a_log"].float() * F.softplus(
            backend.linear(xn, layer["w_alpha"]) + layer["dt_bias"].float()
        ).view(hv)
        # The recurrent step owns q/k L2 norm, the beta sigmoid, the decayed
        # state update and the read.  On CUDA that is one tilelang kernel; the
        # torch backend implements the same contract as the reference.
        read = backend.gdn_decode(q, k, v, g, beta_logits, state.rec, dk**-0.5, cfg.l2_eps)

        z = backend.linear(xn, layer["w_z"]).view(hv, dv)
        normed = backend.rms_norm(read, layer["ssm_norm"], cfg.eps)
        gated = (normed * F.silu(z.float())).reshape(1, -1)
        return backend.linear(gated, layer["w_out"])

    def _mlp(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        layer = self.layers[i]
        backend = self.layer_backends[i]
        gu = backend.linear(xn, layer["w_gu"])
        ffn = self.cfg.ffn
        return backend.linear(backend.silu_mul(gu[:ffn], gu[ffn:]), layer["w_down"])

    @staticmethod
    def _on_device(tensor: torch.Tensor, device: str) -> bool:
        target = torch.device(device)
        if tensor.device.type != target.type:
            return False
        return target.index is None or tensor.device.index == target.index
