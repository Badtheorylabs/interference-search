"""Evaluate every trained model on fresh episodes: success and redundancy at K = 2, 4, 8.

Redundancy = share of dead-end hits that land on a leaf some stream (itself or a
sibling) had already found dead. Mean and spread are over training seeds.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import torch

from env import TreeSearch
from model import StreamPolicy

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs"
DEV = torch.device(os.environ.get("DEVICE", "cpu"))


def eval_model(model, k, args, n=4000, seed=999):
    """Evaluate under the environment settings the model was trained with (`args` from its run file)."""
    g = torch.Generator().manual_seed(seed)
    model.eval()
    env = TreeSearch(n, k, noise=args["noise"], steps=args["steps"], gen=g, refute_p=args["refute_p"], device=DEV)
    env.reset()
    sids = torch.rand(n, 16, generator=g).argsort(1)[:, :k].to(DEV)
    obs = env.observe(torch.zeros(n, k, dtype=torch.bool, device=DEV))
    dead_node = torch.full((n, k), -1, dtype=torch.long, device=DEV)
    buf = {x: [] for x in ("pos_code", "ch_hint", "is_leaf", "dead", "dead_code", "child_code")}
    hits = rep = 0
    for t in range(args["steps"]):
        buf["pos_code"].append(env.code[obs["pos"]])
        buf["ch_hint"].append(obs["ch_hint"])
        buf["is_leaf"].append(obs["is_leaf"])
        buf["dead"].append(obs["dead"])
        buf["dead_code"].append(env.code[dead_node.clamp(min=0)] * (dead_node >= 0).unsqueeze(-1)
                                - (dead_node < 0).unsqueeze(-1).long())
        buf["child_code"].append(env.code[env.children(obs["pos"]).clamp(max=env.n_nodes - 1)])
        inp = {x: torch.stack(v, 2) for x, v in buf.items()}
        with torch.no_grad():
            logits = model(**inp, sids=sids)[:, :, -1]
        a = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]).cpu(), 1, generator=g).view(n, k).to(DEV)
        before = env.dead.clone()
        live = ~env.solved.clone()
        obs, dead_node = env.step(a)
        hit = (dead_node >= 0) & live.unsqueeze(1)
        was = before.gather(1, dead_node.clamp(min=0)) & hit
        hits += hit.sum().item()
        rep += was.sum().item()
    return env.solved.float().mean().item(), rep / max(hits, 1)


res = defaultdict(lambda: defaultdict(list))
strength = defaultdict(list)
for pt in sorted(glob.glob(RUN_DIR + "/*_s*.pt")):
    arm = pt.split("/")[-1].rsplit("_s", 1)[0]
    args = json.load(open(pt.replace(".pt", ".json")))["args"]
    m = StreamPolicy(arm, steps=args["steps"]).to(DEV)
    m.load_state_dict(torch.load(pt, map_location=DEV))
    if arm in ("unsigned", "signed"):
        raw = m.raw.item()
        strength[arm].append(-torch.nn.functional.softplus(torch.tensor(raw)).item() if arm == "signed" else raw)
    for k in (2, 4, 8):
        s, r = eval_model(m, k, args)
        res[arm][f"K{k}"].append(s)
        res[arm][f"R{k}"].append(r)
    print(pt, {k: round(v[-1], 3) for k, v in res[arm].items()}, flush=True)

print()
print("dead-term strength: negative = the pathway lowers logits of children near known dead ends")
print("arm        seeds   K=2            K=4            K=8            redundancy K=4   dead-term strength")
for arm in ("indep", "visible", "unsigned", "signed"):
    if arm not in res:
        continue
    d = res[arm]

    def ms(xs):
        t = torch.tensor(xs)
        return f"{t.mean():.3f}±{t.std(unbiased=True) if len(xs) > 1 else 0:.3f}"
    st = f"{torch.tensor(strength[arm]).mean():+.3f} ({', '.join(f'{x:+.2f}' for x in strength[arm])})" if strength[arm] else ""
    print(f"{arm:9s}  {len(d['K2']):3d}    {ms(d['K2'])}   {ms(d['K4'])}   {ms(d['K8'])}   {ms(d['R4'])}      {st}")
json.dump({a: dict(v) for a, v in res.items()} | {"strength": dict(strength)}, open(RUN_DIR + "/final_eval.json", "w"), indent=1)
