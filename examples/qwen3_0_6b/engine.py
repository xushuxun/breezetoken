"""One captured decode step, replayed.

Ported from TileFoundry examples/qwen3_1_7b-tilelang/fast/engine.py
(MIT, see LICENSE.tilefoundry) with the TileFoundry coupling removed:
weights are loaded straight from the HF safetensors checkpoint and the config
comes from config.json. Architecture: Qwen3ForCausalLM dense decoder, here the
published Qwen3-0.6B, but any Qwen3 dense checkpoint of the same layout works.

Original header follows.

---

The authored reference hands each step's key and value back for the caller to
``torch.cat`` on. That is the right contract for a reference -- it keeps every
shape expressed in ``ctx_len`` alone -- but it means the cache buffer moves
every step, and a graph records addresses, so nothing built that way can be
replayed. This engine takes the other form: a cache of fixed capacity whose
write window advances, with the position in a one-element device tensor.

Everything a step needs then has a fixed address, so the whole step is captured
once and replayed. The chosen token is written back into the input slot by the
last kernel, and while the prompt still has a token left that kernel feeds that
one instead, so the same capture walks the prompt and continues past it with no
host round trip anywhere in the loop.

Weights are repacked once at load: ``q|k|v`` become one matrix and ``gate|up``
another, because a decode GEMV is bandwidth-bound and its cost is the block
count it can fill, not the arithmetic. Two fused reads beat five thin ones.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

import kernels as K

#: Tile shapes per projection, chosen by measuring each under a graph replay:
#: ``(BN, BK, SK, threads)``. ``SK`` is what keeps the SMs busy on the narrow
#: projections -- ``N/BN`` blocks alone leaves most of them idle.
TILES = {
    "qkv": (256, 64, 8, 256),
    "o": (128, 128, 8, 128),
    "gate_up": (256, 64, 2, 256),
    "down": (128, 128, 8, 128),
    "head": (128, 128, 128),
}
#: Context positions one attention block owns. 128, not the upstream 256:
#: 256 positions x 128 dims of K+V blows past this GPU's 99 KB shared-memory
#: opt-in ceiling (AD107); 128 fits with room for the score stage.
SPLIT = 128


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        c = json.loads(Path(path).read_text())
        return cls(
            hidden_size=c["hidden_size"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            intermediate_size=c["intermediate_size"],
            vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            max_position_embeddings=c["max_position_embeddings"],
        )


@dataclass(frozen=True)
class Generated:
    """A continuation and the timing of the part that produced it."""

    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


def _generation_rope(cfg: Config, device: str, cap: int):
    """cos/sin caches, ``cat(freqs, freqs)`` wide -- the pairing the kernel's
    rotate-half reads is ``(d, d + head_dim/2)``."""
    dim = cfg.head_dim
    half = dim // 2
    inverse = cfg.rope_theta ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
    phases = torch.outer(
        torch.arange(cap, device=device, dtype=torch.float32), inverse
    )
    freqs = torch.cat((phases, phases), dim=-1)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


class Engine:
    """A loaded Qwen3 dense model and one captured decode step over it."""

    def __init__(self, ckpt: str | Path, *, device: str = "cuda:0",
                 max_new: int = 256, prompt_room: int = 64):
        ckpt = Path(ckpt)
        self.cfg = Config.from_json(ckpt / "config.json")
        cfg = self.cfg
        self.device = device
        self.dt = torch.bfloat16

        self.H = cfg.hidden_size
        self.HQ = cfg.num_attention_heads
        self.HKV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.I = cfg.intermediate_size
        self.V = cfg.vocab_size
        self.L = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.scale = self.D ** -0.5
        self.qkv_n = self.HQ * self.D + 2 * self.HKV * self.D

        self.nsteps = max_new + prompt_room
        self.cap = ((self.nsteps + SPLIT - 1) // SPLIT) * SPLIT
        self.ns = self.cap // SPLIT

        self._load(ckpt)
        self._buffers()
        self._kernels()
        self._capture()

    # ---------------------------------------------------------------- weights
    def _load(self, ckpt: Path):
        """Fuse the projections a single GEMV can serve, and keep the rest as is."""
        dev, dt = self.device, self.dt

        shards = sorted(ckpt.glob("*.safetensors"))
        tensors = {}
        for s in shards:
            with safe_open(str(s), framework="pt", device=dev) as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)

        def w(name):  # [out,in] -> [in,out], the GEMV layout
            return tensors[name].t().contiguous().to(dt)

        self.w_embed = tensors["model.embed_tokens.weight"].to(dt).contiguous()
        # tied embeddings: Qwen3-0.6B ships no lm_head.weight
        head_name = "lm_head.weight" if "lm_head.weight" in tensors else "model.embed_tokens.weight"
        self.w_head = tensors[head_name].t().contiguous().to(dt)
        self.gamma_final = tensors["model.norm.weight"].to(dt)

        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}.self_attn."
            m = f"model.layers.{i}.mlp."
            self.layers.append({
                "gamma_in": tensors[f"model.layers.{i}.input_layernorm.weight"].to(dt),
                "gamma_post": tensors[f"model.layers.{i}.post_attention_layernorm.weight"].to(dt),
                "gamma_q": tensors[p + "q_norm.weight"].to(dt),
                "gamma_k": tensors[p + "k_norm.weight"].to(dt),
                # [q | k | v] over the output axis: the rope kernel splits it back
                "w_qkv": torch.cat([w(p + "q_proj.weight"),
                                    w(p + "k_proj.weight"),
                                    w(p + "v_proj.weight")], dim=1).contiguous(),
                "w_o": w(p + "o_proj.weight"),
                # [gate | up]: silu_mul consumes both halves of one partial
                "w_gu": torch.cat([w(m + "gate_proj.weight"),
                                   w(m + "up_proj.weight")], dim=1).contiguous(),
                "w_down": w(m + "down_proj.weight"),
            })
        del tensors
        cos, sin = _generation_rope(self.cfg, self.device, self.cap)
        self.cos = cos.contiguous()
        self.sin = sin.contiguous()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- buffers
    def _buffers(self):
        dev, dt, f32 = self.device, self.dt, torch.float32

        def z(*shape, dtype=dt):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        self.hid = z(self.H)
        self.xn = z(self.H)
        self.h1 = z(self.H)
        self.xn1 = z(self.H)
        self.q = z(self.HQ * self.D)
        self.qkv_part = z(TILES["qkv"][2], self.qkv_n, dtype=f32)
        self.o_part = z(TILES["o"][2], self.H, dtype=f32)
        self.gu_part = z(TILES["gate_up"][2], 2 * self.I, dtype=f32)
        self.d_part = z(TILES["down"][2], self.H, dtype=f32)
        self.op = z(self.ns, self.HQ, self.D, dtype=f32)
        self.mp = z(self.ns, self.HQ, dtype=f32)
        self.lp = z(self.ns, self.HQ, dtype=f32)
        self.logits = z(self.V, dtype=f32)
        # A zeroed cache is what makes masking enough: a position past the end
        # contributes exp(-inf) * 0, not exp(-inf) * whatever was there.
        self.kc = z(self.L, self.cap, self.HKV * self.D)
        self.vc = z(self.L, self.cap, self.HKV * self.D)
        self.pos = z(1, dtype=torch.int32)
        self.ids = z(1, dtype=torch.int64)
        self.inp = z(self.nsteps, dtype=torch.int32)
        self.sam = z(self.nsteps, dtype=torch.int32)
        self.plen = z(1, dtype=torch.int32)

    # ---------------------------------------------------------------- kernels
    def _kernels(self):
        H, HQ, HKV, D, I, V = self.H, self.HQ, self.HKV, self.D, self.I, self.V
        self.k_embed = K.embed(V, H)
        self.k_norm = K.rms_norm(H, self.eps)
        self.k_qkv = K.gemv(H, self.qkv_n, *TILES["qkv"])
        self.k_rope = K.qk_rope_cache(
            HQ, HKV, D, self.cap, self.cap, TILES["qkv"][2], self.eps
        )
        self.k_attn = K.attn_partial(HQ, HKV, D, self.cap, SPLIT, self.scale)
        # o_proj merges the attention splits itself; down_proj activates its own
        # slice of the gate/up partial. Two fewer launches per layer.
        self.k_o = K.gemv_attn_combine(HQ, D, H, *TILES["o"][:3], self.ns,
                                       TILES["o"][3])
        self.k_rn_post = K.resid_rms_norm(H, TILES["o"][2], self.eps)
        self.k_gu = K.gemv(H, 2 * I, *TILES["gate_up"])
        self.k_down = K.gemv_silu(I, H, *TILES["down"][:3],
                                  TILES["gate_up"][2], TILES["down"][3])
        self.k_rn_in = K.resid_rms_norm(H, TILES["down"][2], self.eps)
        self.k_head, self.nb = K.lm_head(H, V, *TILES["head"])
        self.bv = torch.zeros(self.nb, device=self.device, dtype=torch.float32)
        self.bi = torch.zeros(self.nb, device=self.device, dtype=torch.int32)
        self.k_sample = K.sample_step(self.nb, self.nsteps)

    def _step(self, record=None):
        """One decode step, as the sequence of launches the graph records."""
        self.k_embed(self.w_embed, self.ids, self.hid)
        self.k_norm(self.hid, self.layers[0]["gamma_in"], self.xn)
        if record is not None:
            record.append(self.hid.clone())
        for i, w in enumerate(self.layers):
            kc, vc = self.kc[i], self.vc[i]
            self.k_qkv(self.xn, w["w_qkv"], self.qkv_part)
            self.k_rope(self.qkv_part, w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                        self.pos, self.pos, kc, vc, self.q)
            self.k_attn(self.q, kc, vc, self.pos, self.op, self.mp, self.lp)
            self.k_o(self.op, self.mp, self.lp, w["w_o"], self.o_part)
            self.k_rn_post(self.hid, self.o_part, w["gamma_post"], self.h1, self.xn1)
            self.k_gu(self.xn1, w["w_gu"], self.gu_part)
            self.k_down(self.gu_part, w["w_down"], self.d_part)
            # The next layer's input norm, or the norm that closes the stack --
            # so the residual add never needs a kernel to itself.
            nxt = (self.layers[i + 1]["gamma_in"] if i + 1 < self.L else self.gamma_final)
            self.k_rn_in(self.h1, self.d_part, nxt, self.hid, self.xn)
            if record is not None:
                record.append(self.hid.clone())
        self.k_head(self.xn, self.w_head, self.logits, self.bv, self.bi)
        self.k_sample(self.bv, self.bi, self.inp, self.plen, self.ids, self.pos, self.sam)

    def _capture(self):
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self._reset()

    def _reset(self):
        self.pos.zero_()
        self.kc.zero_()
        self.vc.zero_()
        self.sam.zero_()

    # ------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new: int) -> Generated:
        """Walk *prompt_ids*, then continue for *max_new* tokens.

        Timing covers exactly the steps that produce the continuation: the step
        at ``pos = len(prompt) - 1`` is the first one whose pick is kept, so the
        steps before it are prefill and are not counted.
        """
        pl = len(prompt_ids)
        if pl < 1:
            raise ValueError("decode needs a prompt of at least one token")
        if pl + max_new > self.nsteps:
            raise ValueError(
                f"prompt {pl} + {max_new} new exceeds this engine's {self.nsteps} steps"
            )
        self._reset()
        self.inp.zero_()
        self.inp[:pl] = torch.tensor(prompt_ids, device=self.device, dtype=torch.int32)
        self.plen.fill_(pl)
        self.ids.fill_(prompt_ids[0])

        torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        for _ in range(pl - 1):                     # prefill: picks discarded
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        t1 = perf_counter()
        for _ in range(max_new):                    # the continuation itself
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        t2 = perf_counter()

        out = self.sam[pl - 1: pl - 1 + max_new].tolist()
        return Generated(out, t2 - t1, t1 - t0, pl)
