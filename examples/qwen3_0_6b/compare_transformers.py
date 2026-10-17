"""Compare the ported engine against transformers (HF reference) numerics.

For each prompt: logits at the last prompt position (max abs diff, top-1
agreement, top-5 overlap) and a greedy continuation diff position-by-position.
Run:  uv run --with transformers python examples/qwen3_0_6b/compare_transformers.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

CKPT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots"
CKPT = next(CKPT.iterdir())

PROMPTS = [
    "The capital of France is",
    "Write a short essay about why the sky is blue, in three paragraphs.",
    "def quicksort(arr):",
    "1 + 1 = 2, 2 + 2 = 4, 4 + 4 =",
]
MAX_NEW = 32

USE_FP32_REFERENCE = "--fp32-ref" in sys.argv


def main() -> int:
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    print("loading HF reference (bf16, eager attention)...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        str(CKPT), torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to("cuda:0")

    hf32 = None
    if USE_FP32_REFERENCE:
        print("loading fp32 reference...", flush=True)
        hf32 = AutoModelForCausalLM.from_pretrained(
            str(CKPT), torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval().to("cuda:0")

    import engine as E
    if "--tileops" in sys.argv:
        import engine_tileops as E
    room = max(len(tok.encode(p, add_special_tokens=False).ids) for p in PROMPTS) + 8
    eng = (E.EngineTileOps if "--tileops" in sys.argv else E.Engine)(
        CKPT, device="cuda:0", max_new=MAX_NEW, prompt_room=room)

    print(f"{'prompt':50s} {'max|dlogit|':>11s} {'top1':>5s} {'top5':>5s} {'greedy match'}")
    all_ok = True
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False).ids
        with torch.no_grad():
            hf_logits = hf(input_ids=torch.tensor([ids], device="cuda:0")).logits[0, -1].float()
            hf_out = hf.generate(
                input_ids=torch.tensor([ids], device="cuda:0"),
                max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=0,
            )[0, len(ids):].tolist()

        my_logits = eng.logits_for(ids)
        d = (hf_logits - my_logits).abs()
        hf_top1 = hf_logits.argmax().item()
        my_top1 = my_logits.argmax().item()
        hf_top5 = set(hf_logits.topk(5).indices.tolist())
        my_top5 = set(my_logits.topk(5).indices.tolist())

        dist = ""
        if hf32 is not None:
            with torch.no_grad():
                ref = hf32(input_ids=torch.tensor([ids], device="cuda:0")).logits[0, -1].float()
            dist = (f"  |d(fp32): hf={ (hf_logits - ref).abs().max():.4f}"
                    f" mine={(my_logits - ref).abs().max():.4f}")

        mine = eng.generate(ids, MAX_NEW).tokens
        match = sum(a == b for a, b in zip(hf_out, mine))
        status = "OK" if (d.max() < 0.5 and match >= MAX_NEW - 2) else "CHECK"
        all_ok &= status == "OK"
        print(f"{p[:50]:50s} {d.max():11.4f} "
              f"{'Y' if hf_top1 == my_top1 else 'N':>5s} {len(hf_top5 & my_top5):>5d} "
              f"{match}/{MAX_NEW} {status}{dist}")

        if hf_top1 != my_top1:
            print(f"    hf top1={hf_top1} ({tok.decode([hf_top1])!r})  "
                  f"mine={my_top1} ({tok.decode([my_top1])!r})")

    print("RESULT:", "ALL CONSISTENT" if all_ok else "SEE CHECK ROWS ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
