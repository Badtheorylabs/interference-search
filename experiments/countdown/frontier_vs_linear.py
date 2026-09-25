"""Linear vs frontier at matched compute on unseen Countdown problems, with the same trained judge.

Reports the solve rate within each expansion budget and the sequential rounds used, for a linear chain,
a linear chain that abandons judged-dead states, a merged frontier and a merged frontier with a hard
judge cut-off. Add --ablation for the frontier without merging and best-first search.
"""
import argparse
import json
import random

import torch

from interference_search.countdown import gen_problem
from interference_search.judge import load_judge
from searches import bestfirst, frontier, linear

ap = argparse.ArgumentParser()
ap.add_argument("--n6", type=int, default=100)
ap.add_argument("--n7", type=int, default=30)
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--ablation", action="store_true")
ap.add_argument("--out", default="frontier_vs_linear.json")
args = ap.parse_args()

judge = load_judge()
BUDGETS = (10, 20, 50, 100, 200, 400)
arms = {
    "linear": lambda p, b, rng: linear(judge, p["numbers"], p["target"], b, rng),
    "linear + refute": lambda p, b, rng: linear(judge, p["numbers"], p["target"], b, rng, refute_tau=args.tau),
    "frontier": lambda p, b, rng: frontier(judge, p["numbers"], p["target"], b),
    "frontier + refute": lambda p, b, rng: frontier(judge, p["numbers"], p["target"], b, refute_tau=args.tau),
}
if args.ablation:
    arms.update({
        "frontier, no merge": lambda p, b, rng: frontier(judge, p["numbers"], p["target"], b, merge=False),
        "best-first": lambda p, b, rng: bestfirst(judge, p["numbers"], p["target"], b),
        "best-first, no merge": lambda p, b, rng: bestfirst(judge, p["numbers"], p["target"], b, merge=False),
    })

out = {}
for n, count, seed0 in ((6, args.n6, 40_000), (7, args.n7, 50_000)):
    probs = [gen_problem(random.Random(seed0 + i), n_numbers=n) for i in range(count)]
    print(f"\n{n}-number unseen problems ({count}); solve rate by expansion budget "
          + " ".join(f"{b:>5d}" for b in BUDGETS), flush=True)
    out[n] = {}
    for name, fn in arms.items():
        rng = torch.Generator().manual_seed(0)
        rates, rounds = [], 0.0
        for b in BUDGETS:
            res = [fn(p, b, rng) for p in probs]
            rates.append(sum(r[0] for r in res) / len(probs))
            ok = [r for r in res if r[0]]
            rounds = sum(r[2] for r in ok) / max(len(ok), 1)
        out[n][name] = {"solve": dict(zip(BUDGETS, rates)), "rounds_at_max_budget": rounds}
        print(f"  {name:22s} " + "  ".join(f"{x:5.0%}" for x in rates) + f"    rounds {rounds:6.1f}", flush=True)
json.dump(out, open(args.out, "w"), indent=1)
