"""Qwen3.5 (qwen35): hybrid Gated DeltaNet + GQA decoder.

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
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from breezetoken.backends import Backend, make_backend
from breezetoken.gguf import GGUFFile

#: Qwen3.5 text config defaults for the 0.8B checkpoint; every value is
#: overridden from GGUF metadata in ``Qwen35Config.from_gguf``.
_DEFAULT_CONTEXT = 8192


@dataclass(frozen=True)
class Qwen35Config:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    ffn: int
    vocab: int
    eps: float
    l2_eps: float
    ssm_inner: int
    ssm_state: int
    ssm_heads: int
    conv_k: int
    full_attention_interval: int
    max_context: int
    nextn_layers: int = 0

    @property
    def attn_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if (i + 1) % self.full_attention_interval == 0)

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, max_context: int | None = None) -> Qwen35Config:
        md = gguf.metadata
        n_heads = int(md["qwen35.attention.head_count"])
        head_dim = int(
            md.get("qwen35.attention.key_length", int(md["qwen35.embedding_length"]) // n_heads)
        )
        nextn = int(md.get("qwen35.nextn_predict_layers", 0))
        n_layers = int(md["qwen35.block_count"]) - nextn
        context_length = int(md.get("qwen35.context_length", _DEFAULT_CONTEXT))
        requested = _DEFAULT_CONTEXT if max_context is None else int(max_context)
        if requested <= 0:
            raise ValueError("max_context must be positive")
        context = min(context_length, requested) if context_length > 0 else requested
        cfg = cls(
            hidden=int(md["qwen35.embedding_length"]),
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv=int(md["qwen35.attention.head_count_kv"]),
            head_dim=head_dim,
            rope_dim=int(md.get("qwen35.rope.dimension_count", head_dim)),
            rope_theta=float(md["qwen35.rope.freq_base"]),
            ffn=int(md["qwen35.feed_forward_length"]),
            vocab=int(gguf.tensor_info("token_embd.weight")[0][0]),
            eps=float(md.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6)),
            l2_eps=1e-6,
            ssm_inner=int(md["qwen35.ssm.inner_size"]),
            ssm_state=int(md["qwen35.ssm.state_size"]),
            ssm_heads=int(md["qwen35.ssm.group_count"]),
            conv_k=int(md["qwen35.ssm.conv_kernel"]),
            full_attention_interval=int(md.get("qwen35.full_attention_interval", 4)),
            max_context=context,
            nextn_layers=nextn,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.full_attention_interval <= 0:
            raise ValueError("full_attention_interval must be positive")
        if not self.attn_layers:
            raise ValueError("model has no full-attention layers")
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError(f"invalid rope_dim {self.rope_dim}")
        if self.n_heads % self.n_kv:
            raise ValueError("n_heads must be divisible by n_kv")
        if self.ssm_heads * self.ssm_state != self.ssm_inner:
            raise ValueError("ssm group_count * state_size must equal ssm.inner_size")


class KVCache:
    """Full-attention KV cache, preallocated for ``max_context`` positions."""

    def __init__(self, cfg: Qwen35Config, n_attn: int, device: str, dtype: torch.dtype):
        shape = (n_attn, cfg.max_context, cfg.n_kv * cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def reset(self) -> None:
        # Positions are overwritten before they are read, so clearing is not
        # required; keep the method for an explicit lifecycle.
        pass


class GDNState:
    """Per-GDN-layer convolution window and recurrent state."""

    def __init__(self, cfg: Qwen35Config, device: str):
        self.conv = torch.zeros(
            3 * cfg.ssm_inner, cfg.conv_k - 1, device=device, dtype=torch.float32
        )
        self.rec = torch.zeros(
            cfg.ssm_heads, cfg.ssm_state, cfg.ssm_state, device=device, dtype=torch.float32
        )

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()


class Qwen35:
    """A loaded Qwen3.5 decoder and its per-token forward pass."""

    def __init__(
        self,
        gguf: GGUFFile,
        *,
        backend: str = "auto",
        device: str = "auto",
        dtype: str | torch.dtype | None = None,
        max_context: int | None = None,
    ) -> None:
        self.cfg = Qwen35Config.from_gguf(gguf, max_context)
        self.backend: Backend = make_backend(backend, device, dtype)
        self.device = self.backend.device
        self.dtype = self.backend.weight_dtype
        self.w: dict[str, torch.Tensor] = {}
        self.layers: list[dict[str, torch.Tensor]] = []
        self._load(gguf)
        self._validate_shapes()

        self._attn_layers = self.cfg.attn_layers
        self._attn_set = frozenset(self._attn_layers)
        self._attn_index = {layer: i for i, layer in enumerate(self._attn_layers)}
        self.kv = KVCache(self.cfg, len(self._attn_layers), self.device, torch.float32)
        self.states = {
            i: GDNState(self.cfg, self.device)
            for i in range(self.cfg.n_layers)
            if i not in self._attn_set
        }
        self._cos, self._sin = self._rope_tables(self.cfg.max_context)
        self._pos_buf = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._sdpa_gqa = True

    # ------------------------------------------------------------------ load
    def _weight(self, name: str) -> torch.Tensor:
        """2-D weight in (K, N) GEMV layout, in the backend's weight dtype."""
        tensor = self.gguf_get(name, self.dtype)
        if tensor.dim() != 2:
            raise ValueError(f"expected a 2-D weight for {name}, got {tuple(tensor.shape)}")
        return tensor.t().contiguous()

    def gguf_get(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        tensor = self._gguf.get_tensor(name, dtype=dtype)
        return tensor.to(self.device)

    def _norm(self, name: str) -> torch.Tensor:
        """Norm gammas stay f32 even when projection weights are bf16."""
        return self.gguf_get(name, torch.float32)

    def _raw(self, name: str) -> torch.Tensor:
        """A non-GEMV tensor (conv kernel) in f32."""
        return self.gguf_get(name, torch.float32)

    def _load(self, gguf: GGUFFile) -> None:
        self._gguf = gguf
        # Tied embeddings: keep one (H, V) matrix and read a column for the
        # token lookup.  The old code kept both (V, H) and (H, V), doubling
        # the embedding memory for no benefit.
        self.w["head"] = self._weight("token_embd.weight")
        self.w["norm_f"] = self._norm("output_norm.weight")
        for i in range(self.cfg.n_layers):
            prefix = f"blk.{i}."
            w_gate = self._weight(prefix + "ffn_gate.weight")
            w_up = self._weight(prefix + "ffn_up.weight")
            layer = {
                "norm_in": self._norm(prefix + "attn_norm.weight"),
                "norm_post": self._norm(prefix + "post_attention_norm.weight"),
                # One GEMV produces gate and up; the MLP splits the result.
                "w_gu": torch.cat((w_gate, w_up), dim=1).contiguous(),
                "w_down": self._weight(prefix + "ffn_down.weight"),
            }
            del w_gate, w_up
            if i in self.cfg.attn_layers:
                w_q = self._weight(prefix + "attn_q.weight")
                w_k = self._weight(prefix + "attn_k.weight")
                w_v = self._weight(prefix + "attn_v.weight")
                layer.update(
                    # One GEMV for q (with its output gate), k and v.
                    w_qkv_attn=torch.cat((w_q, w_k, w_v), dim=1).contiguous(),
                    w_o=self._weight(prefix + "attn_output.weight"),
                    q_norm=self._norm(prefix + "attn_q_norm.weight"),
                    k_norm=self._norm(prefix + "attn_k_norm.weight"),
                )
                del w_q, w_k, w_v
            else:
                layer.update(
                    w_qkv=self._weight(prefix + "attn_qkv.weight"),
                    w_z=self._weight(prefix + "attn_gate.weight"),
                    conv=self._raw(prefix + "ssm_conv1d.weight"),
                    w_alpha=self._weight(prefix + "ssm_alpha.weight"),
                    w_beta=self._weight(prefix + "ssm_beta.weight"),
                    a_log=self._norm(prefix + "ssm_a"),
                    dt_bias=self._norm(prefix + "ssm_dt.bias"),
                    w_out=self._weight(prefix + "ssm_out.weight"),
                    ssm_norm=self._norm(prefix + "ssm_norm.weight"),
                )
            self.layers.append(layer)
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

    # ----------------------------------------------------------------- pieces
    def _embed(self, token: int) -> torch.Tensor:
        if token < 0 or token >= self.cfg.vocab:
            raise ValueError(f"token {token} out of range [0, {self.cfg.vocab})")
        # head is (H, V): a column is the token embedding.  The copy is 4 KB
        # and keeps the strided read out of the hot residual path.
        return self.w["head"][:, token].float().contiguous()

    def _rope_tables(self, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        half = cfg.rope_dim // 2
        inv = cfg.rope_theta ** (
            -torch.arange(0, half, device=self.device, dtype=torch.float32) / half
        )
        phases = torch.outer(torch.arange(cap, device=self.device, dtype=torch.float32), inv)
        freqs = torch.cat((phases, phases), dim=-1)
        return freqs.cos(), freqs.sin()

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        rows = x.numel() // dim
        out = self.backend.rope(
            x.reshape(rows, dim).contiguous(),
            self._cos,
            self._sin,
            self._pos_buf,
            self.cfg.rope_dim,
        )
        return out.reshape(x.shape)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Grouped-query decode attention; q (1,H,1,D), k/v (1,Hkv,S,D)."""
        try:
            if self._sdpa_gqa:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.cfg.attn_scale,
                    enable_gqa=True,
                )
        except (RuntimeError, TypeError):
            self._sdpa_gqa = False
        cfg = self.cfg
        group = cfg.n_heads // cfg.n_kv
        qg = q.reshape(cfg.n_kv, group, cfg.head_dim)
        scores = torch.einsum("kgd,ksd->kgs", qg, k[0]) * cfg.attn_scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("kgs,ksd->kgd", probs, v[0])
        return out.reshape(1, cfg.n_heads, 1, cfg.head_dim)

    def _attn(self, i: int, xn: torch.Tensor, pos: int) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        head_slice = self._attn_index[i]

        qkv = self.backend.linear(xn, layer["w_qkv_attn"])
        q_size = cfg.n_heads * 2 * cfg.head_dim
        kv_size = cfg.n_kv * cfg.head_dim
        qg = qkv[:q_size].view(1, cfg.n_heads, 2 * cfg.head_dim)
        q, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
        k = qkv[q_size : q_size + kv_size].view(1, cfg.n_kv, cfg.head_dim)
        v = qkv[q_size + kv_size :].view(1, cfg.n_kv, cfg.head_dim)

        q = self._rope(self.backend.rms_norm(q, layer["q_norm"], cfg.eps))
        k = self._rope(self.backend.rms_norm(k, layer["k_norm"], cfg.eps))

        self.kv.k[head_slice, pos] = k.reshape(-1)
        self.kv.v[head_slice, pos] = v.reshape(-1)
        seq = pos + 1
        kk = self.kv.k[head_slice, :seq].view(1, seq, cfg.n_kv, cfg.head_dim).transpose(1, 2)
        vv = self.kv.v[head_slice, :seq].view(1, seq, cfg.n_kv, cfg.head_dim).transpose(1, 2)

        out = self._attend(q.view(1, cfg.n_heads, 1, cfg.head_dim), kk, vv)
        out = out.view(1, cfg.n_heads, cfg.head_dim) * torch.sigmoid(gate.float())
        return self.backend.linear(out.reshape(1, -1), layer["w_o"])

    def _gdn(self, i: int, xn: torch.Tensor, state: GDNState) -> torch.Tensor:
        cfg = self.cfg
        layer = self.layers[i]
        hv, dk, dv = cfg.ssm_heads, cfg.ssm_state, cfg.ssm_state

        entry = self.backend.linear(xn, layer["w_qkv"])
        window = torch.cat((state.conv, entry.reshape(-1, 1)), dim=1)
        mixed = F.silu((window * layer["conv"]).sum(dim=-1))
        state.conv.copy_(window[:, 1:])

        q = mixed[: cfg.ssm_inner].view(hv, dk)
        k = mixed[cfg.ssm_inner : 2 * cfg.ssm_inner].view(hv, dk)
        v = mixed[2 * cfg.ssm_inner :].view(hv, dv)

        beta_logits = self.backend.linear(xn, layer["w_beta"]).view(hv)
        # GGUF ssm_a is already -exp(A_log); do not exp() it again.
        g = layer["a_log"].float() * F.softplus(
            self.backend.linear(xn, layer["w_alpha"]) + layer["dt_bias"].float()
        ).view(hv)
        # The recurrent step owns q/k L2 norm, the beta sigmoid, the decayed
        # state update and the read.  On CUDA that is one tilelang kernel; the
        # torch backend implements the same contract as the reference.
        read = self.backend.gdn_decode(q, k, v, g, beta_logits, state.rec, dk**-0.5, cfg.l2_eps)

        z = self.backend.linear(xn, layer["w_z"]).view(hv, dv)
        normed = self.backend.rms_norm(read, layer["ssm_norm"], cfg.eps)
        gated = (normed * F.silu(z.float())).reshape(1, -1)
        return self.backend.linear(gated, layer["w_out"])

    def _mlp(self, i: int, xn: torch.Tensor) -> torch.Tensor:
        layer = self.layers[i]
        gu = self.backend.linear(xn, layer["w_gu"])
        ffn = self.cfg.ffn
        return self.backend.linear(self.backend.silu_mul(gu[:ffn], gu[ffn:]), layer["w_down"])

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
        self._pos_buf.fill_(pos)
        h = self._embed(token).view(1, -1)
        # The residual add and the norm that follows it are one read, so every
        # layer boundary uses fused_add_rms_norm.  The norm after the last MLP
        # is norm_f and produces the head input directly.
        xn = self.backend.rms_norm(h, self.layers[0]["norm_in"], cfg.eps)
        for i in range(cfg.n_layers):
            layer = self.layers[i]
            if i in self._attn_set:
                mixer = self._attn(i, xn, pos)
            else:
                mixer = self._gdn(i, xn, self.states[i])
            h, xn1 = self.backend.fused_add_rms_norm(h, mixer, layer["norm_post"], cfg.eps)
            mlp = self._mlp(i, xn1)
            next_norm = self.layers[i + 1]["norm_in"] if i + 1 < cfg.n_layers else self.w["norm_f"]
            h, xn = self.backend.fused_add_rms_norm(h, mlp, next_norm, cfg.eps)
        if not logits:
            return None
        return self.backend.linear(xn, self.w["head"]).float()

    def reset(self) -> None:
        self.kv.reset()
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
        """Compile the kernels for every shape used by one decode step."""
        self.reset()
        self.generate([0], 1, 0.0)
        self.reset()

    def close(self) -> None:
        self.backend.close()

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"Qwen35(hidden={cfg.hidden}, layers={cfg.n_layers}, "
            f"attn={len(self._attn_layers)}, max_context={cfg.max_context}, "
            f"backend={self.backend.name}, dtype={self.dtype})"
        )
