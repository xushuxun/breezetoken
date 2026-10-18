"""Batched prompt prefill for the native Qwen3.5 model.

Projections, causal attention and the dense MLP run over the whole prompt;
the GDN delta recurrence is inherently sequential and calls the same decode
kernel once per token.  This is deliberately separate from the token-wise
decode path in ``qwen35.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from breezetoken.backends import Backend


class PrefillMixin:
    # ------------------------------------------------------------------ prefill
    def _rope_batch(self, x: torch.Tensor, steps: int, backend: Backend) -> torch.Tensor:
        cfg = self.cfg
        device = backend.device
        self._ensure_rope_state(device)
        cos, sin = self._rope_tables_by_device[device]
        rot = x[..., : cfg.rope_dim].float()
        tail = x[..., cfg.rope_dim :].float()
        half = cfg.rope_dim // 2
        c = cos[:steps].view(steps, 1, cfg.rope_dim)
        s = sin[:steps].view(steps, 1, cfg.rope_dim)
        turned = rot * c + torch.cat((-rot[..., half:], rot[..., :half]), dim=-1) * s
        return torch.cat((turned, tail), dim=-1).to(x.dtype)

    @torch.inference_mode()
    def _prefill_attn(self, i: int, xn: torch.Tensor, steps: int) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        backend = self.layer_backends[i]
        cache = self.kv[i]

        qkv = backend.linear_batch(xn, layer["w_qkv_attn"])
        q_size = cfg.n_heads * 2 * cfg.head_dim
        kv_size = cfg.n_kv * cfg.head_dim
        qg = qkv[:, :q_size].reshape(steps, cfg.n_heads, 2 * cfg.head_dim)
        q, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
        k = qkv[:, q_size : q_size + kv_size].reshape(steps, cfg.n_kv, cfg.head_dim)
        v = qkv[:, q_size + kv_size :].reshape(steps, cfg.n_kv, cfg.head_dim)

        q = backend.rms_norm(q.reshape(steps * cfg.n_heads, cfg.head_dim), layer["q_norm"], cfg.eps)
        k = backend.rms_norm(k.reshape(steps * cfg.n_kv, cfg.head_dim), layer["k_norm"], cfg.eps)
        q = self._rope_batch(q.reshape(steps, cfg.n_heads, cfg.head_dim), steps, backend)
        k = self._rope_batch(k.reshape(steps, cfg.n_kv, cfg.head_dim), steps, backend)

        cache.k[:steps] = k.reshape(steps, -1)
        cache.v[:steps] = v.reshape(steps, -1)
        qq = q.transpose(0, 1).unsqueeze(0)
        kk = k.transpose(0, 1).unsqueeze(0)
        vv = v.transpose(0, 1).unsqueeze(0)
        try:
            out = F.scaled_dot_product_attention(
                qq,
                kk,
                vv,
                dropout_p=0.0,
                is_causal=True,
                scale=cfg.attn_scale,
                enable_gqa=True,
            )
        except (RuntimeError, TypeError):
            group = cfg.n_heads // cfg.n_kv
            qg = q.reshape(steps, cfg.n_kv, group, cfg.head_dim).permute(1, 2, 0, 3)
            scores = torch.einsum("kgtd,ksd->kgts", qg, k) * cfg.attn_scale
            mask = torch.ones(steps, steps, device=xn.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            out = torch.einsum("kgts,ksd->kgtd", torch.softmax(scores, dim=-1), v)
            out = out.permute(2, 0, 1, 3).reshape(steps, cfg.n_heads, cfg.head_dim)
            out = out.transpose(0, 1).unsqueeze(0)
        out = out[0].transpose(0, 1) * torch.sigmoid(gate.float())
        return backend.linear_batch(out.reshape(steps, cfg.n_heads * cfg.head_dim), layer["w_o"])

    @torch.inference_mode()
    def _prefill_gdn(self, i: int, xn: torch.Tensor, steps: int) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        backend = self.layer_backends[i]
        state = self.states[i]
        hv, dk, dv = cfg.ssm_heads, cfg.ssm_state, cfg.ssm_state
        channels = 3 * cfg.ssm_inner

        entry = backend.linear_batch(xn, layer["w_qkv"])
        seq = torch.cat((state.conv, entry.t()), dim=1).unsqueeze(0)
        mixed = F.silu(F.conv1d(seq, layer["conv"].unsqueeze(1), groups=channels)).squeeze(0).t()
        state.conv.copy_(seq[0, :, -3:])

        q = mixed[:, : cfg.ssm_inner].reshape(steps, hv, dk)
        k = mixed[:, cfg.ssm_inner : 2 * cfg.ssm_inner].reshape(steps, hv, dk)
        v = mixed[:, 2 * cfg.ssm_inner :].reshape(steps, hv, dv)
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + cfg.l2_eps)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + cfg.l2_eps)

        beta = backend.linear_batch(xn, layer["w_beta"])
        g = layer["a_log"].float() * F.softplus(
            backend.linear_batch(xn, layer["w_alpha"]) + layer["dt_bias"].float()
        )
        read = torch.empty((steps, hv, dv), dtype=torch.float32, device=xn.device)
        for t in range(steps):
            read[t] = backend.gdn_decode(
                q[t], k[t], v[t], g[t], beta[t], state.rec, dk**-0.5, cfg.l2_eps
            )

        z = backend.linear_batch(xn, layer["w_z"]).reshape(steps, hv, dv)
        normed = backend.rms_norm(read.reshape(steps * hv, dv), layer["ssm_norm"], cfg.eps).reshape(
            steps, hv, dv
        )
        gated = (normed * F.silu(z)).reshape(steps, cfg.ssm_inner)
        return backend.linear_batch(gated, layer["w_out"])

    @torch.inference_mode()
    def _prefill_mlp(self, i: int, xn: torch.Tensor, steps: int) -> torch.Tensor:
        layer = self.layers[i]
        backend = self.layer_backends[i]
        ffn = self.cfg.ffn
        gu = backend.linear_batch(xn, layer["w_gu"])
        gated = backend.silu_mul(gu[:, :ffn].reshape(-1), gu[:, ffn:].reshape(-1))
        return backend.linear_batch(gated.reshape(steps, ffn), layer["w_down"])

    @torch.inference_mode()
    def _prefill_batch(self, ids: Sequence[int]) -> torch.Tensor:
        cfg = self.cfg
        steps = len(ids)
        tokens = torch.tensor(list(ids), dtype=torch.long, device=self.head_device)
        h = self.w["head"].index_select(1, tokens).t().float().contiguous()
        xn: torch.Tensor | None = None
        for i in range(cfg.n_layers):
            backend = self.layer_backends[i]
            layer = self.layers[i]
            device = backend.device
            if not self._on_device(h, device):
                h = h.to(device)
            if xn is None or not self._on_device(xn, device):
                xn = backend.rms_norm(h, layer["norm_in"], cfg.eps)
            if i in self._attn_set:
                mixer = self._prefill_attn(i, xn, steps)
            else:
                mixer = self._prefill_gdn(i, xn, steps)
            h, xn1 = backend.fused_add_rms_norm(h, mixer, layer["norm_post"], cfg.eps)
            mlp = self._prefill_mlp(i, xn1, steps)
            h, xn = backend.fused_add_rms_norm(h, mlp, layer["norm_next"], cfg.eps)
        if not self._on_device(xn, self.head_device):
            xn = xn.to(self.head_device)
        return self.head_backend.linear(xn[-1], self.w["head"]).float()
