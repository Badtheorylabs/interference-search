"""Measure real per-step compute for the lab models: forward+backward FLOPs in
train mode (avoids the fused attention fast path that FlopCounterMode cannot
see), the interference unit's share, and GPU wall time per training step."""
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose import build_model, loss_of
from envs import make_dataset
from model import param_count
from run_lab import BS, DEVICE, build_rows


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def main():
    torch.manual_seed(0)
    (ids, tgt), _ = build_rows(make_dataset(seed=0))
    ids, tgt = ids[:BS].to(DEVICE), tgt[:BS].to(DEVICE)
    res = {}
    for mode in ["vanilla", "interference"]:
        torch.manual_seed(0)
        m = build_model(mode).train()
        step = lambda: loss_of(m, mode, ids, tgt)
        fwd = flops(step)
        fwd_bwd = flops(lambda: step().backward())
        unit_fwd = None
        if mode == "interference":
            def trunk():
                x = m.tok(ids)
                for b in m.blocks:
                    x = b(x)
            unit_fwd = fwd - flops(trunk)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for _ in range(5):
            opt.zero_grad(); step().backward(); opt.step()
        sync(); t0 = time.time()
        for _ in range(50):
            opt.zero_grad(); step().backward(); opt.step()
        sync()
        res[mode] = dict(fwd=fwd, fwd_bwd=fwd_bwd, unit_fwd=unit_fwd,
                         ms_per_step=(time.time() - t0) / 50 * 1000, params=param_count(m))
        print(mode, res[mode])
    v, i = res["vanilla"], res["interference"]
    print(f"FLOPS_RATIO interference/vanilla (fwd+bwd, train mode): {i['fwd_bwd'] / v['fwd_bwd']:.3f}")
    print(f"WALL_RATIO  interference/vanilla (ms per step): {i['ms_per_step'] / v['ms_per_step']:.2f}")
    print(f"UNIT_SHARE of interference forward FLOPs: {i['unit_fwd'] / i['fwd']:.3f}")


if __name__ == "__main__":
    main()
