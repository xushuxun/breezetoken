"""Three-way decode benchmark: TileFoundry-graph vs TileOPs-eager vs transformers.

Each implementation runs in its OWN process -- the eager engine allocates
output tensors per kernel call, and a co-resident HF model fragments the CUDA
caching-allocator pool enough to make those allocations sync (7x slowdown).
Separate processes are the fair and stable methodology.

Run: uv run --with transformers python examples/qwen3_0_6b/bench_3way.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots"
CKPT = next(CKPT.iterdir())

SHORT = "The capital of France is"
LONG = ("Write a detailed explanation of how a GPU executes a matrix multiplication, "
        "covering the roles of SMs, warps, shared memory and tensor cores. " * 8)
MAX_NEW = 128

CHILD = r'''
import os, sys, torch
from pathlib import Path
from time import perf_counter
sys.path.insert(0, %r)
from tokenizers import Tokenizer

CKPT = Path(%r)
tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
ids_s = tok.encode(%r, add_special_tokens=False).ids
ids_l = tok.encode(%r, add_special_tokens=False).ids
impl = %r
n = %d

import statistics

def med_engine(Engine, ids, new, **kw):
    """Warm up once (covers all JIT paths), then time 3 full runs; return
    (median prefill s, median decode s/token)."""
    eng = Engine(CKPT, max_new=new, prompt_room=len(ids) + 8, **kw)
    eng.generate(ids, new)  # warm up
    pres, decs = [], []
    for _ in range(3):
        r = eng.generate(ids, new)
        pres.append(r.prefill_seconds)
        decs.append(r.seconds / new)
    return statistics.median(pres), statistics.median(decs)

if impl == "cpu":
    import engine_cpu as M
    pre_s, dec_s = med_engine(M.EngineCPU, ids_s, n)
    pre_l, dec_l = med_engine(M.EngineCPU, ids_l, n)
    dec_s = 1 / dec_s; dec_l = 1 / dec_l
elif impl == "hybrid":
    import engine_hybrid as M
    pre_s, dec_s = med_engine(M.EngineHybrid, ids_s, n, device_map="alternating")
    pre_l, dec_l = med_engine(M.EngineHybrid, ids_l, n, device_map="alternating")
    dec_s = 1 / dec_s; dec_l = 1 / dec_l
elif impl == "hf":
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(str(CKPT), torch_dtype=torch.bfloat16,
                                              attn_implementation="eager").eval().to("cuda:0")
    def run(ids, new):
        x = torch.tensor([ids], device="cuda:0")
        torch.cuda.synchronize(); t0 = perf_counter()
        out = hf.generate(x, max_new_tokens=new, do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        return perf_counter() - t0
    import statistics
    run(ids_s, 4)  # warm up
    def pair(ids, new):
        t1, tn = run(ids, 1), run(ids, new)
        dec = (tn - t1) / (new - 1)
        return t1 - dec, dec
    pres, decs = zip(*[pair(ids_s, n) for _ in range(3)])
    pre_s, dec_s = statistics.median(pres), statistics.median(decs)
    prel, decl = zip(*[pair(ids_l, n) for _ in range(3)])
    pre_l, dec_l = statistics.median(prel), statistics.median(decl)
    dec_s = 1 / dec_s; dec_l = 1 / dec_l
else:
    if impl == "graph":
        import engine as M
        Engine = M.Engine
    else:
        import engine_tileops as M
        Engine = M.EngineTileOps
    pre_s, dec_s = med_engine(Engine, ids_s, n)
    pre_l, dec_l = med_engine(Engine, ids_l, n)
    dec_s = 1 / dec_s; dec_l = 1 / dec_l

print(f"RESULT {impl} {1e3*pre_s:.2f} {dec_s:.1f} {1e3*pre_l:.2f} {dec_l:.1f}")
'''


def bench(impl: str) -> tuple[float, float, float, float]:
    code = CHILD % (str(HERE), str(CKPT), SHORT, LONG, impl, MAX_NEW)
    out = subprocess.run(
        ["uv", "run", "--with", "transformers", "python", "-c", code],
        capture_output=True, text=True, cwd=str(HERE.parent.parent), timeout=1200,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            _, _, pre_s, dec_s, pre_l, dec_l = line.split()
            return float(pre_s), float(dec_s), float(pre_l), float(dec_l)
    print(out.stdout[-2000:], out.stderr[-2000:], file=sys.stderr)
    raise SystemExit(f"bench failed for {impl}")


def main() -> int:
    print(f"five-way, per-implementation subprocesses, median of 3 runs, greedy {MAX_NEW} new tokens; "
          "decode tok/s excludes prefill", flush=True)
    rows = {}
    for impl, label in (("graph", "engine.py (TileFoundry + CUDA graph)"),
                        ("tileops", "engine_tileops.py (TileOPs components, CUDA graph)"),
                        ("hybrid", "engine_hybrid.py (CPU-GPU alternating)"),
                        ("cpu", "engine_cpu.py (tilelang CPU target)"),
                        ("hf", "transformers (HF bf16, eager attn)")):
        pre_s, dec_s, pre_l, dec_l = bench(impl)
        rows[label] = (pre_s, dec_s, pre_l, dec_l)
        print(f"  {label}: done", flush=True)

    print(f"\n{'implementation':44s} {'short prompt(5 tok)':>22s} {'long prompt(233 tok)':>22s}")
    print(f"{'':44s} {'prefill ms':>10s} {'decode tok/s':>11s} {'prefill ms':>10s} {'decode tok/s':>11s}")
    for label, (pre_s, dec_s, pre_l, dec_l) in rows.items():
        print(f"{label:44s} {pre_s:10.1f} {dec_s:11.1f} {pre_l:10.1f} {dec_l:11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
