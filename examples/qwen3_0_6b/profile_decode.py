"""Per-stage decode breakdown for the GPU eager path, the graph engine, and the
CPU engine -- the numbers behind the hardware-roofline explanation.

Run: uv run python examples/qwen3_0_6b/profile_decode.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

CKPT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots"
CKPT = next(CKPT.iterdir())
POS = 96           # fixed context length for the breakdown
REPS = 60


def profile_tileops() -> dict:
    """engine_tileops pieces, driven eagerly (same kernels the graph replays)."""
    import engine_tileops as ET
    eng = ET.EngineTileOps(CKPT, device="cuda:0", max_new=256, prompt_room=256)
    g = eng
    g.pos.fill_(POS)
    g.ids.fill_(42)

    stages = {}

    def timeit(name, fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        stages[name] = (perf_counter() - t0) / REPS * 1e3

    w0 = g.layers[0]
    xn = torch.randn(g.H, device="cuda:0", dtype=torch.bfloat16)
    resid = torch.randn(g.H, device="cuda:0", dtype=torch.bfloat16)

    timeit("embed", lambda: g.k_embed(g.w_embed, g.ids, g.hid))
    timeit("rms_norm(tops)", lambda: g.t_norm.forward(g.hid, w0["gamma_in"]))
    def qkv():
        g.k_qkv(xn, w0["w_qkv"], g.qkv_part)
    timeit("qkv_gemv", qkv)
    def rope():
        g.k_rope(g.qkv_part, w0["gamma_q"], w0["gamma_k"], g.cos, g.sin,
                 g.pos, g.pos, g.kc[0], g.vc[0], g.q)
    timeit("qk_rope_cache", rope)
    def attn():
        g.k_attn_t(g.q.view(1, g.HQ, g.D),
                   g.kc[0].view(1, g.cap, g.HKV, g.D),
                   g.vc[0].view(1, g.cap, g.HKV, g.D),
                   g.pos, g.attn_buf.view(1, g.HQ, g.D))
    timeit("attn_masked(tops)", attn)
    def oproj():
        g.k_o(g.attn_buf, w0["w_o"], g.o_part)
    timeit("o_gemv", oproj)
    o_sum = g.o_part.sum(0).to(torch.bfloat16)
    timeit("fused_post(tops)", lambda: g.t_fused.forward(o_sum, resid, w0["gamma_post"]))
    def gu():
        g.k_gu(xn, w0["w_gu"], g.gu_part)
    timeit("gate_up_gemv", gu)
    def down():
        g.k_down(g.gu_part, w0["w_down"], g.d_part)
    timeit("down_gemv(fused silu)", down)
    d_sum = g.d_part.sum(0).to(torch.bfloat16)
    timeit("fused_next(tops)", lambda: g.t_fused.forward(d_sum, resid, w0["gamma_in"]))
    def head():
        g.k_head(xn, g.w_head, g.logits, g.bv, g.bi)
    timeit("lm_head+argmax", head)
    return stages, g.cap


def profile_graph() -> float:
    import engine as E
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    ids = tok.encode("The capital of France is", add_special_tokens=False).ids
    eng = E.Engine(CKPT, device="cuda:0", max_new=256, prompt_room=256)
    eng.generate(ids, 16)  # warm
    torch.cuda.synchronize()
    t0 = perf_counter()
    n = 200
    for _ in range(n):
        eng.graph.replay()
    torch.cuda.synchronize()
    return (perf_counter() - t0) / n * 1e3


def profile_cpu() -> dict:
    import engine_cpu as EC
    eng = EC.EngineCPU(CKPT, max_new=256, prompt_room=256)
    c = eng
    c.pos.fill_(POS)
    c.ids.fill_(42)
    stages = {}

    def timeit(name, fn):
        for _ in range(3):
            fn()
        t0 = perf_counter()
        for _ in range(REPS // 3):
            fn()
        stages[name] = (perf_counter() - t0) / (REPS // 3) * 1e3

    w0 = c.layers[0]
    xn = torch.randn(c.H)
    resid = torch.randn(c.H)

    timeit("embed", lambda: c.k_embed(c.w_embed, c.ids, c.hid))
    timeit("rms_norm", lambda: c.k_norm(c.hid, w0["gamma_in"], c.xn))
    timeit("qkv_gemv", lambda: c.k_qkv(xn, w0["w_qkv"], c.qkv_buf))
    timeit("qk_rope_cache", lambda: c.k_rope(c.qkv_buf, w0["gamma_q"], w0["gamma_k"],
                                             c.cos, c.sin, c.pos, c.kc[0], c.vc[0], c.q))
    timeit("attn", lambda: c.k_attn(c.q, c.kc[0], c.vc[0], c.pos, c.attn_out))
    timeit("o_gemv", lambda: c.k_o(c.attn_out, w0["w_o"], c.obuf))
    timeit("fused_post", lambda: c.k_fused(resid, c.obuf, w0["gamma_post"], c.h1, c.xn1))
    timeit("gate_up_gemv", lambda: c.k_gu(xn, w0["w_gu"], c.gu))
    timeit("silu_mul", lambda: c.k_silu(c.gu, c.act))
    timeit("down_gemv", lambda: c.k_down(c.act, w0["w_down"], c.dout))
    timeit("fused_next", lambda: c.k_fused(c.h1, c.dout, w0["gamma_in"], c.hid, c.xn))
    timeit("lm_head", lambda: c.k_head(xn, c.w_head, c.logits))
    return stages, c.cap


def main() -> int:
    print(f"per-stage breakdown at pos={POS} (ms/stage-call)\n")

    g_stages, g_cap = profile_tileops()
    print(f"[engine_tileops pieces] (cap={g_cap}; x28 layers for per-layer stages)")
    tot_layer = 0.0
    for name, ms in g_stages.items():
        per_tok = ms * 28 if name not in ("embed", "lm_head+argmax") else ms
        tot_layer += per_tok
        print(f"  {name:24s} {ms:7.4f} ms   x28 -> {per_tok:7.3f} ms/token")
    print(f"  {'SUM (eager, incl launch)':24s} {'':7s}    {tot_layer:7.3f} ms/token")

    ms = profile_graph()
    print(f"\n[engine.py] whole-step graph replay: {ms:.3f} ms/token")

    c_stages, c_cap = profile_cpu()
    print(f"\n[engine_cpu] (cap={c_cap}; f32 weights, single thread)")
    tot = 0.0
    for name, ms in c_stages.items():
        per_tok = ms * 28 if name not in ("embed", "lm_head") else ms
        tot += per_tok
        print(f"  {name:24s} {ms:8.3f} ms   x28 -> {per_tok:8.1f} ms/token")
    print(f"  {'SUM (kernel time)':24s} {'':8s}    {tot:8.1f} ms/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
