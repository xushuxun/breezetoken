"""Continue a prompt with Qwen3-0.6B on the ported TileFoundry engine, and report the rate.

    uv run examples/qwen3_0_6b/run.py --prompt "..." --max-new-tokens 256

Adapted from TileFoundry examples/qwen3_1_7b-tilelang/run.py (MIT): the
tokenizer comes from the `tokenizers` library instead of transformers, and the
device is this machine's single GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CKPT = Path.home() / ".cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots"
CKPT = next(CKPT.iterdir()) if CKPT.is_dir() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="prompt to continue")
    ap.add_argument("--max-new-tokens", type=int, default=256, metavar="TOKENS")
    ap.add_argument("--ckpt", type=Path, default=CKPT,
                    help="HF checkpoint directory (default: modelscope cache)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer
    import engine as E

    tokenizer = Tokenizer.from_file(str(args.ckpt / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise SystemExit("decode needs a prompt that encodes to at least one token")

    t0 = perf_counter()
    eng = E.Engine(args.ckpt, device=args.device,
                   max_new=args.max_new_tokens, prompt_room=len(prompt_ids) + 8)
    build = perf_counter() - t0

    result = eng.generate(prompt_ids, args.max_new_tokens)
    text = tokenizer.decode(result.tokens)

    print(text)
    print()
    print(f"{result.tokens_per_second:.1f} tok/s "
          f"({len(result.tokens)} tokens in {result.seconds:.3f}s, "
          f"{1e3 * result.seconds / len(result.tokens):.3f} ms/token)")
    print(f"  device {args.device}  |  prompt {result.prompt_steps} tokens, "
          f"prefill {1e3 * result.prefill_seconds:.1f}ms  |  "
          f"load+compile {build:.1f}s  |  {len(text)} characters generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
