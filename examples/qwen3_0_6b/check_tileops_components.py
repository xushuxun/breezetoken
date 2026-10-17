"""Component-level checks: TileOPs-derived kernels vs torch references.

Run: uv run python examples/qwen3_0_6b/check_tileops_components.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

DT = torch.bfloat16
DEV = "cuda:0"
H = 1024
EPS = 1e-6


def rms_ref(x, w, eps=EPS):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(DT).float() * w.float()


def gqa_ref(q, kk, vv):
    HQ, D = q.shape[2], q.shape[3]
    S, HKV = kk.shape[1], kk.shape[2]
    G = HQ // HKV
    qf = q.float().reshape(HQ, 1, D) * (D ** -0.5)
    kf = kk.float().reshape(S, HKV, D).repeat_interleave(G, dim=1).permute(1, 2, 0)
    vf = vv.float().reshape(S, HKV, D).repeat_interleave(G, dim=1).permute(1, 0, 2)
    probs = torch.softmax(torch.matmul(qf, kf), dim=-1)
    return torch.matmul(probs, vf).reshape(HQ, D)


def main() -> int:
    torch.manual_seed(0)
    ok = True

    # ---------------- rms_norm ----------------
    from tileops_lite.kernels.norm.rms_norm import RMSNormKernel
    k = RMSNormKernel(H, EPS, DT)
    x = torch.randn(H, device=DEV, dtype=DT)
    w = torch.randn(H, device=DEV, dtype=DT)
    y = k.forward(x, w)
    d = (y.float() - rms_ref(x, w)).abs().max().item()
    print(f"rms_norm:        max|diff|={d:.5f}")
    ok &= d < 0.02

    # ---------------- fused add + rms_norm ----------------
    from tileops_lite.kernels.norm.fused_add_norm import FusedAddRMSNormKernel
    k = FusedAddRMSNormKernel(H, EPS, DT)
    a = torch.randn(H, device=DEV, dtype=DT)
    b = torch.randn(H, device=DEV, dtype=DT)
    g = torch.randn(H, device=DEV, dtype=DT)
    y, res = k.forward(a, b, g)
    ref_res = (a.float() + b.float()).to(DT)
    d1 = (res.float() - ref_res.float()).abs().max().item()
    d2 = (y.float() - rms_ref(ref_res, g)).abs().max().item()
    print(f"fused_add_norm:  residual max|diff|={d1:.5f}  norm max|diff|={d2:.5f}")
    ok &= d1 < 0.02 and d2 < 0.02

    # ---------------- GQA decode (no-split and split paths) ----------------
    from tileops_lite.kernels.attention.gqa.decode import GQADecodeKernel
    HQ, HKV, D = 16, 8, 128
    for seqlen in (1, 37, 128, 700):
        k = GQADecodeKernel(1, HQ, HKV, seqlen, D, "bfloat16")
        q = torch.randn(1, 1, HQ, D, device=DEV, dtype=DT)
        kk = torch.randn(1, seqlen, HKV, D, device=DEV, dtype=DT)
        vv = torch.randn(1, seqlen, HKV, D, device=DEV, dtype=DT)
        o = k.forward(q, kk, vv).float().reshape(HQ, D)
        ref = gqa_ref(q, kk, vv)
        d = (o - ref).abs().max().item()
        print(f"gqa_decode S={seqlen:4d}: max|diff|={d:.5f}")
        ok &= d < 0.05

    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
