"""Qwen3.5 (qwen35): model facade, loading and generation.

The model consumes one token at a time (decode).  It is backend-agnostic; the
``Backend`` object owns kernel selection and the torch fallback, so the same
model code runs as a pure-torch reference or through tilelang kernels.

Numerical conventions verified against transformers:

* GGUF norm gammas are pre-folded (HF stores ``w`` and computes ``(1 + w)``);
* ``ssm_a`` stores ``-exp(A_log)``, so the log-decay is ``a * softplus(...)``;
* the GGUF ``block_count`` includes the MTP block;
* Qwen3.5 uses partial RoPE on the leading ``rope_dim`` dims;
* the attention projection is fused ``[q | output-gate]`` per head.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace

import torch

from breezetoken.backends import Backend, make_backend
from breezetoken.gguf import GGUFFile
from breezetoken.model.config import GDNState, KVCache, Qwen35Config
from breezetoken.model.graph import GraphMixin
from breezetoken.model.layers import LayerMixin
from breezetoken.model.prefill import PrefillMixin
from breezetoken.planner import Hardware, Placement, plan

#: Prompts at least this long take the batched prefill path.
_PREFILL_BATCH_MIN = 8


class Qwen35(LayerMixin, PrefillMixin, GraphMixin):
    """A loaded Qwen3.5 decoder and its per-token forward pass.

    Auto placement keeps a GPU prefix long enough to fit the weights, KV and
    GDN state in VRAM; the remaining suffix runs on the CPU through torch.  The
    residual stream crosses devices once at the boundary.  When everything fits
    (the 0.8B checkpoint on an 8GB card), auto is all-GPU.
    """

    def __init__(
        self,
        gguf: GGUFFile,
        *,
        backend: str = "auto",
        device: str = "auto",
        dtype: str | torch.dtype | None = None,
        max_context: int | None = None,
        max_gpu_layers: int | None = None,
    ) -> None:
        cfg = Qwen35Config.from_gguf(gguf, max_context)
        auto = backend == "auto" and device == "auto"
        if auto:
            placement = plan(
                cfg,
                Hardware.detect(),
                max_context=max_context,
                max_gpu_layers=max_gpu_layers,
            )
            if placement.max_context != cfg.max_context:
                cfg = replace(cfg, max_context=placement.max_context)
                cfg.validate()
            gpu_backend: Backend | None = None
            cpu_backend: Backend | None = None
            if placement.gpu_layers > 0 or placement.head_on_gpu:
                gpu_backend = make_backend("tilelang", "cuda", dtype="bfloat16")
            if placement.gpu_layers < cfg.n_layers or not placement.head_on_gpu:
                cpu_backend = make_backend("torch", "cpu", dtype="float32")
            backends: list[Backend] = [
                gpu_backend if i < placement.gpu_layers else cpu_backend
                for i in range(cfg.n_layers)
            ]
            head_backend = gpu_backend if placement.head_on_gpu else cpu_backend
            if head_backend is None or any(b is None for b in backends):
                raise RuntimeError("planner produced an incomplete backend assignment")
            self.gpu_backend = gpu_backend
            self.cpu_backend = cpu_backend
        else:
            single = make_backend(backend, device, dtype)
            backends = [single] * cfg.n_layers
            head_backend = single
            on_gpu = single.device.startswith("cuda")
            placement = Placement(
                max_context=cfg.max_context,
                gpu_layers=cfg.n_layers if on_gpu else 0,
                head_on_gpu=on_gpu,
                gpu_bytes=0,
                cpu_bytes=0,
                reason=f"explicit {single.name}/{single.device}",
            )
            self.gpu_backend = single if on_gpu else None
            self.cpu_backend = single if not on_gpu else None

        self.cfg = cfg
        self.placement = placement
        self.layer_backends: list[Backend] = backends
        self.head_backend: Backend = head_backend
        self.backend = self.head_backend  # compatibility for callers/repr
        self.head_device = self.head_backend.device
        self.head_dtype = self.head_backend.weight_dtype
        self.device = self.head_device
        self.dtype = self.head_dtype
        self._unique_backends = list(
            {id(b): b for b in [*self.layer_backends, self.head_backend]}.values()
        )

        self.w: dict[str, torch.Tensor] = {}
        self.layers: list[dict[str, torch.Tensor]] = []
        self._load(gguf)
        self._validate_shapes()

        self._attn_layers = self.cfg.attn_layers
        self._attn_set = frozenset(self._attn_layers)
        self.kv = {
            i: KVCache(self.cfg, self.layer_backends[i].device, torch.float32)
            for i in self._attn_layers
        }
        self.states = {
            i: GDNState(self.cfg, self.layer_backends[i].device)
            for i in range(self.cfg.n_layers)
            if i not in self._attn_set
        }
        self._rope_tables_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pos_bufs: dict[str, torch.Tensor] = {}
        for dev in sorted({b.device for b in self.layer_backends} | {self.head_device}):
            self._ensure_rope_state(dev)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_error: str | None = None
        self._graph_pos: torch.Tensor | None = None
        self._graph_index: torch.Tensor | None = None
        self._graph_token: torch.Tensor | None = None
        self._graph_capable = bool(
            placement.gpu_layers == cfg.n_layers
            and placement.head_on_gpu
            and all(
                b.name == "tilelang" and b.device.startswith("cuda") for b in self.layer_backends
            )
            and cfg.n_heads == 8
            and cfg.n_kv == 2
            and cfg.head_dim == 256
        )

    # ------------------------------------------------------------------ load
    def _weight(self, name: str, backend: Backend) -> torch.Tensor:
        """2-D weight in (K, N) GEMV layout, on this layer's device/dtype."""
        tensor = self.gguf_get(name, backend.weight_dtype, backend.device)
        if tensor.dim() != 2:
            raise ValueError(f"expected a 2-D weight for {name}, got {tuple(tensor.shape)}")
        return tensor.t().contiguous()

    def gguf_get(self, name: str, dtype: torch.dtype, device: str) -> torch.Tensor:
        return self._gguf.get_tensor(name, dtype=dtype).to(device)

    def _norm(self, name: str, device: str) -> torch.Tensor:
        """Norm gammas stay f32 even when projection weights are bf16."""
        return self.gguf_get(name, torch.float32, device)

    def _raw(self, name: str, device: str) -> torch.Tensor:
        """A non-GEMV tensor (conv kernel) in f32 on the layer device."""
        return self.gguf_get(name, torch.float32, device)

    def _load(self, gguf: GGUFFile) -> None:
        self._gguf = gguf
        # Tied embeddings: one (H, V) matrix, read as a column for lookup.
        self.w["head"] = self._weight("token_embd.weight", self.head_backend)
        attn = set(self.cfg.attn_layers)
        for i in range(self.cfg.n_layers):
            backend = self.layer_backends[i]
            device = backend.device
            prefix = f"blk.{i}."
            w_gate = self._weight(prefix + "ffn_gate.weight", backend)
            w_up = self._weight(prefix + "ffn_up.weight", backend)
            layer = {
                "norm_in": self._norm(prefix + "attn_norm.weight", device),
                "norm_post": self._norm(prefix + "post_attention_norm.weight", device),
                "w_gu": torch.cat((w_gate, w_up), dim=1).contiguous(),
                "w_down": self._weight(prefix + "ffn_down.weight", backend),
            }
            del w_gate, w_up
            if i in attn:
                w_q = self._weight(prefix + "attn_q.weight", backend)
                w_k = self._weight(prefix + "attn_k.weight", backend)
                w_v = self._weight(prefix + "attn_v.weight", backend)
                layer.update(
                    w_qkv_attn=torch.cat((w_q, w_k, w_v), dim=1).contiguous(),
                    w_o=self._weight(prefix + "attn_output.weight", backend),
                    q_norm=self._norm(prefix + "attn_q_norm.weight", device),
                    k_norm=self._norm(prefix + "attn_k_norm.weight", device),
                )
                del w_q, w_k, w_v
            else:
                layer.update(
                    w_qkv=self._weight(prefix + "attn_qkv.weight", backend),
                    w_z=self._weight(prefix + "attn_gate.weight", backend),
                    conv=self._raw(prefix + "ssm_conv1d.weight", device),
                    w_alpha=self._weight(prefix + "ssm_alpha.weight", backend),
                    w_beta=self._weight(prefix + "ssm_beta.weight", backend),
                    a_log=self._norm(prefix + "ssm_a", device),
                    dt_bias=self._norm(prefix + "ssm_dt.bias", device),
                    w_out=self._weight(prefix + "ssm_out.weight", backend),
                    ssm_norm=self._norm(prefix + "ssm_norm.weight", device),
                )
            self.layers.append(layer)

        # The norm after the last MLP is norm_f.  Each layer keeps its next
        # norm weight on its own device, so the fused kernel never has to
        # touch a device boundary.
        norm_f = self._norm("output_norm.weight", self.layer_backends[-1].device)
        self.w["norm_f"] = norm_f
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1]["norm_in"] if i + 1 < self.cfg.n_layers else norm_f
            if nxt.device != self.layer_backends[i].device:
                nxt = nxt.to(self.layer_backends[i].device)
            layer["norm_next"] = nxt
        del self._gguf

    def _validate_shapes(self) -> None:
        cfg = self.cfg
        d = self.w["head"]
        if d.shape != (cfg.hidden, cfg.vocab):
            raise ValueError(f"token_embd shape {tuple(d.shape)} != {(cfg.hidden, cfg.vocab)}")
        for i, layer in enumerate(self.layers):
            expect = {
                "w_gu": (cfg.hidden, 2 * cfg.ffn),
                "w_down": (cfg.ffn, cfg.hidden),
            }
            if i in cfg.attn_layers:
                expect.update(
                    w_qkv_attn=(
                        cfg.hidden,
                        cfg.n_heads * 2 * cfg.head_dim + 2 * cfg.n_kv * cfg.head_dim,
                    ),
                    w_o=(cfg.n_heads * cfg.head_dim, cfg.hidden),
                )
            else:
                expect.update(
                    w_qkv=(cfg.hidden, 3 * cfg.ssm_inner),
                    w_z=(cfg.hidden, cfg.ssm_inner),
                    w_out=(cfg.ssm_inner, cfg.hidden),
                )
            for name, shape in expect.items():
                got = tuple(layer[name].shape)
                if got != shape:
                    raise ValueError(f"blk.{i}.{name}: expected {shape}, got {got}")

    # ------------------------------------------------------------------ step
    @torch.inference_mode()
    def forward_token(self, token: int, pos: int, *, logits: bool = True):
        """Consume one token at absolute position ``pos``.

        Returns next-token logits only when ``logits=True``.  Prefill skips the
        248k-vocabulary head for every prompt token except the last.
        """
        cfg = self.cfg
        if not 0 <= pos < cfg.max_context:
            raise ValueError(f"position {pos} outside [0, {cfg.max_context})")
        for buf in self._pos_bufs.values():
            buf.fill_(pos)
        h = self._embed(token).view(1, -1)
        # Cross a device boundary only when the placement changes; with a
        # contiguous GPU prefix that is at most once per token.
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
                mixer = self._attn(i, xn, pos)
            else:
                mixer = self._gdn(i, xn, self.states[i])
            h, xn1 = backend.fused_add_rms_norm(h, mixer, layer["norm_post"], cfg.eps)
            mlp = self._mlp(i, xn1)
            h, xn = backend.fused_add_rms_norm(h, mlp, layer["norm_next"], cfg.eps)
        if not logits:
            return None
        if not self._on_device(xn, self.head_device):
            xn = xn.to(self.head_device)
        return self.head_backend.linear(xn, self.w["head"]).float()

    def reset(self) -> None:
        for cache in self.kv.values():
            cache.reset()
        for state in self.states.values():
            state.reset()

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

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

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
        """Yield generated token ids until EOS, cancellation, or max_new.

        Logits are computed for the last prompt token and for each generated
        token; prompt tokens before the last skip the vocabulary head.
        """
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

        self.reset()
        eos = frozenset(eos_ids)
        generator = self._make_generator(seed) if temperature > 0 else None

        logits = None
        if len(ids) >= _PREFILL_BATCH_MIN:
            if cancel is not None and cancel.is_set():
                return
            logits = self._prefill_batch(ids)
            if cancel is not None and cancel.is_set():
                return
        else:
            last_prompt = len(ids) - 1
            for i, token in enumerate(ids):
                if cancel is not None and cancel.is_set():
                    return
                logits = self.forward_token(token, i, logits=(i == last_prompt))
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
            if step == 0 and self._graph_eligible(temperature, top_p):
                token = self._graph_step(nxt, len(ids))
                if token is not None:
                    for gstep in range(1, max_new):
                        if cancel is not None and cancel.is_set():
                            return
                        if token in eos:
                            return
                        yield token
                        if gstep + 1 >= max_new:
                            return
                        if cancel is not None and cancel.is_set():
                            return
                        token = self._graph_replay()
                    return
                # Capture failed; finish this step eagerly and stay on the
                # normal path for the rest of the sequence.
                logits = self.forward_token(nxt, len(ids), logits=True)
            else:
                logits = self.forward_token(nxt, len(ids) + step, logits=True)

    def generate(
        self,
        ids: Sequence[int],
        max_new: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[int]:
        return list(self.generate_stream(ids, max_new, temperature, **kwargs))

    def warmup(self) -> None:
        """Compile kernels and capture the graph for one decode step."""
        self.reset()
        self.generate([0], 1, 0.0)
        self.reset()
        if self._graph_capable and self._graph is None and self.cfg.max_context > 1:
            self.forward_token(0, 0, logits=True)
            self._graph_step(0, 1)
            self.reset()

    def close(self) -> None:
        for backend in self._unique_backends:
            backend.close()

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"Qwen35(hidden={cfg.hidden}, layers={cfg.n_layers}, "
            f"attn={len(self._attn_layers)}, max_context={cfg.max_context}, "
            f"{self.placement.summary(cfg.n_layers)}, dtype={self.dtype})"
        )


__all__ = ["GDNState", "KVCache", "Qwen35", "Qwen35Config"]
