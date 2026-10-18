"""CUDA graph capture for the all-GPU greedy decode path.

The graph is captured once (warmup or first long generation) and replayed for
later tokens.  Everything that changes between tokens -- position, token id,
KV writes -- is a device tensor inside the graph; the host only reads the next
token.  Hybrid placement, sampling and unsupported attention shapes fall back
to the eager path in ``qwen35.py``.
"""

from __future__ import annotations

import contextlib

import torch


class GraphMixin:
    # ------------------------------------------------------------------ graph
    def _ensure_graph_buffers(self) -> None:
        if self._graph_pos is not None:
            return
        device = self.head_device
        self._graph_pos = torch.zeros(1, dtype=torch.int32, device=device)
        self._graph_index = torch.zeros(1, dtype=torch.int64, device=device)
        self._graph_token = torch.zeros(1, dtype=torch.int64, device=device)
        # RoPE reads the same int32 position buffer that the graph increments.
        self._pos_bufs[device] = self._graph_pos

    @torch.inference_mode()
    def _graph_embed(self) -> torch.Tensor:
        assert self._graph_token is not None
        return self.w["head"].index_select(1, self._graph_token).t().float().contiguous()

    @torch.inference_mode()
    def _graph_attn(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        backend = self.layer_backends[i]
        cache = self.kv[i]

        qkv = backend.linear(xn, layer["w_qkv_attn"])
        q_size = cfg.n_heads * 2 * cfg.head_dim
        kv_size = cfg.n_kv * cfg.head_dim
        qg = qkv[:q_size].view(1, cfg.n_heads, 2 * cfg.head_dim)
        q, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
        k = qkv[q_size : q_size + kv_size].view(1, cfg.n_kv, cfg.head_dim)
        v = qkv[q_size + kv_size :].view(1, cfg.n_kv, cfg.head_dim)
        q = self._rope(backend.rms_norm(q, layer["q_norm"], cfg.eps), backend)
        k = self._rope(backend.rms_norm(k, layer["k_norm"], cfg.eps), backend)

        assert self._graph_index is not None and self._graph_pos is not None
        cache.k.index_copy_(0, self._graph_index, k.reshape(1, -1))
        cache.v.index_copy_(0, self._graph_index, v.reshape(1, -1))
        cap = cfg.max_context
        out = backend.attention_fixed(
            q.view(cfg.n_heads, cfg.head_dim),
            cache.k.view(cap, cfg.n_kv, cfg.head_dim),
            cache.v.view(cap, cfg.n_kv, cfg.head_dim),
            self._graph_pos,
            cfg.attn_scale,
        )
        out = out.view(1, cfg.n_heads, cfg.head_dim) * torch.sigmoid(gate.float())
        return backend.linear(out.reshape(1, -1), layer["w_o"])

    @torch.inference_mode()
    def _graph_forward(self) -> torch.Tensor:
        cfg = self.cfg
        h = self._graph_embed()
        backend = self.layer_backends[0]
        xn = backend.rms_norm(h, self.layers[0]["norm_in"], cfg.eps)
        for i in range(cfg.n_layers):
            layer = self.layers[i]
            backend = self.layer_backends[i]
            if i in self._attn_set:
                mixer = self._graph_attn(i, xn)
            else:
                mixer = self._gdn(i, xn, self.states[i])
            h, xn1 = backend.fused_add_rms_norm(h, mixer, layer["norm_post"], cfg.eps)
            mlp = self._mlp(i, xn1)
            h, xn = backend.fused_add_rms_norm(h, mlp, layer["norm_next"], cfg.eps)
        logits = self.head_backend.linear(xn, self.w["head"]).float()
        assert self._graph_pos is not None and self._graph_index is not None
        self._graph_pos.add_(1)
        self._graph_index.add_(1)
        return logits

    @torch.inference_mode()
    def _graph_body(self) -> None:
        logits = self._graph_forward()
        assert self._graph_token is not None
        self._graph_token.copy_(logits.argmax().reshape(1))

    def _graph_preflight(self) -> None:
        cfg = self.cfg
        device = self.head_device
        q = torch.zeros(cfg.n_heads, cfg.head_dim, device=device)
        for i in self._attn_layers:
            cache = self.kv[i]
            self.layer_backends[i].attention_fixed(
                q,
                cache.k.view(cfg.max_context, cfg.n_kv, cfg.head_dim),
                cache.v.view(cfg.max_context, cfg.n_kv, cfg.head_dim),
                self._graph_pos,
                cfg.attn_scale,
            )
        torch.cuda.synchronize()

    def _graph_step(self, first_token: int, pos: int) -> int | None:
        if not self._graph_capable or self._graph_error is not None:
            return None
        self._ensure_graph_buffers()
        assert (
            self._graph_pos is not None
            and self._graph_index is not None
            and self._graph_token is not None
        )
        self._graph_pos.fill_(pos)
        self._graph_index.fill_(pos)
        self._graph_token.fill_(first_token)
        if self._graph is None:
            try:
                self._graph_preflight()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._graph_body()
            except Exception as exc:  # noqa: BLE001
                self._graph_error = f"{type(exc).__name__}: {exc}"
                self._graph_capable = False
                self._graph = None
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
                return None
            self._graph = graph
        self._graph.replay()
        return int(self._graph_token.item())

    def _graph_replay(self) -> int:
        assert self._graph is not None and self._graph_token is not None
        self._graph.replay()
        return int(self._graph_token.item())

    def _graph_eligible(self, temperature: float, top_p: float) -> bool:
        return (
            self._graph_capable
            and self._graph_error is None
            and temperature == 0.0
            and top_p == 1.0
        )
