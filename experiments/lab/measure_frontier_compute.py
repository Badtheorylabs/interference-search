"""FLOPs per training example (fwd+bwd) and wall time per step for the
frontier ladder, so the vanilla baseline can be sized to matched compute."""
import os
import random
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_env as E
from frontier_model import VanillaReach
from run_frontier import BS, DEVICE, LADDER, build, loss_of


def measure(model, name, batch, gen):
    model.train()
    with FlopCounterMode(display=False) as fc:
        loss_of(model, name, batch, gen).backward()
    flops = fc.get_total_flops() / len(batch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        opt.zero_grad(); loss_of(model, name, batch, gen).backward(); opt.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        opt.zero_grad(); loss_of(model, name, batch, gen).backward(); opt.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return flops, (time.time() - t0) / 10 * 1000


def main():
    rng = random.Random(0)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    train, _ = E.split_starts(0)
    batch = E.sample(E.by_target(train), BS, rng)
    for name in LADDER:
        torch.manual_seed(0)
        m = build(name)
        f, ms = measure(m, name, batch, gen)
        print(f"{name:14s} params {sum(p.numel() for p in m.parameters()):>9,d}  MFLOPs/example {f / 1e6:8.1f}  ms/step {ms:6.1f}")
    for d, layers in [(128, 2), (256, 2), (256, 4), (384, 4), (512, 4), (512, 6)]:
        torch.manual_seed(0)
        m = VanillaReach(d=d, layers=layers).to(DEVICE)
        f, ms = measure(m, "vanilla", batch, gen)
        print(f"vanilla d={d} L={layers}  params {sum(p.numel() for p in m.parameters()):>9,d}  MFLOPs/example {f / 1e6:8.1f}  ms/step {ms:6.1f}")


if __name__ == "__main__":
    main()
