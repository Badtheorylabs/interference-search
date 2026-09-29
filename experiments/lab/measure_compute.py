"""Measure real per-step compute for the lab models: forward+backward FLOPs in
train mode (avoids the fused attention fast path that FlopCounterMode cannot
see), the interference unit's share, and GPU wall time per training step."""
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import make_dataset
from model import InterferenceLM, VanillaLM, match_params, param_count
from run_lab import BS, D, DEVICE, K, L, VOCAB, build_rows, ce_state

KW = dict(prop_mode="learned", merge_mode="learned", survival_mode="learned")


def build(mode):
    target = param_count(InterferenceLM(VOCAB, D, L, k=K))
    if mode == "vanilla":
        return match_params(VanillaLM(VOCAB, D, L, n_out=90), target).to(DEVICE)
    return InterferenceLM(VOCAB, D, L, k=K).to(DEVICE)


def loss_of(m, mode, ids, tgt):
    valid = ids != 0
    if mode == "vanilla":
        return ce_state(m(ids, valid=valid).view(-1, 9, 10), tgt)
    return ce_state(m(ids, valid=valid, **KW)[2], tgt)


def main():
    torch.manual_seed(0)
    (ids, tgt), _ = build_rows(make_dataset(seed=0))
    ids, tgt = ids[:BS].to(DEVICE), tgt[:BS].to(DEVICE)
    res = {}
    for mode in ["vanilla", "interference"]:
        torch.manual_seed(0)
        m = build(mode).train()
        with FlopCounterMode(display=False) as fc:
            loss = loss_of(m, mode, ids, tgt)
        fwd = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc2:
            loss = loss_of(m, mode, ids, tgt)
            loss.backward()
        total = fc2.get_total_flops()
        unit_flops = None
        if mode == "interference":
            with FlopCounterMode(display=False) as fc3:
                x = m.tok(ids)
                for b in m.blocks:
                    x = b(x)
            unit_flops = fwd - fc3.get_total_flops()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for _ in range(5):
            opt.zero_grad(); loss_of(m, mode, ids, tgt).backward(); opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            opt.zero_grad(); loss_of(m, mode, ids, tgt).backward(); opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        res[mode] = dict(fwd=fwd, fwd_bwd=total, unit_fwd=unit_flops,
                         ms_per_step=(time.time() - t0) / 50 * 1000, params=param_count(m))
        print(mode, res[mode])
    r = res["interference"]["fwd_bwd"] / res["vanilla"]["fwd_bwd"]
    w = res["interference"]["ms_per_step"] / res["vanilla"]["ms_per_step"]
    print(f"FLOPS_RATIO interference/vanilla (fwd+bwd, train mode): {r:.3f}")
    print(f"WALL_RATIO  interference/vanilla (ms per step): {w:.2f}")
    print(f"UNIT_SHARE of interference forward FLOPs: {res['interference']['unit_fwd'] / res['interference']['fwd']:.3f}")


if __name__ == "__main__":
    main()
