"""Isolate the cause of slot collapse at init: which operator makes all K
frontier slots identical? Runs the unit step by step with operators toggled."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import make_dataset
from model import InterferenceLM
from run_lab import D, DEVICE, K, L, VOCAB, build_rows


def slot_cos(f):
    fn = F.normalize(f, dim=-1)
    c = fn @ fn.transpose(1, 2)
    off = ~torch.eye(f.size(1), dtype=torch.bool, device=f.device)
    return c[:, off].mean().item()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    (ids, _), _ = build_rows(make_dataset(seed=0))
    ids = ids[:64].to(DEVICE)
    m = InterferenceLM(VOCAB, D, L, k=K).to(DEVICE).eval()
    x = m.tok(ids)
    for b in m.blocks:
        x = b(x)
    f0 = m.slot_init.unsqueeze(0).expand(64, K, D).contiguous()
    print(f"slot_init norm {f0.norm(dim=-1).mean():.3f}   trunk token norm {x[:, 0].norm(dim=-1).mean():.3f}")
    print(f"slot cosine before any step: {slot_cos(f0):.4f}")
    xc = x[:, 0].unsqueeze(1).expand(64, K, D)
    p = m.unit.prop(torch.cat([f0, xc], -1))
    print(f"proposal delta norm {p.norm(dim=-1).mean():.3f}; cosine between slots' proposal deltas: {slot_cos(p):.4f}")
    for prop, merge, surv in [("learned", "learned", "learned"), ("off", "learned", "learned"),
                              ("learned", "off", "learned"), ("off", "off", "learned"),
                              ("off", "off", "one")]:
        f = f0.clone()
        for t in range(8):
            f, _, _ = m.unit(f, x[:, t], prop, merge, surv)
        print(f"prop={prop:7s} merge={merge:7s} survival={surv:7s} -> slot cosine after 8 steps {slot_cos(f):.4f}, norm {f.norm(dim=-1).mean():.3f}")


if __name__ == "__main__":
    main()
