"""The assembled system on Countdown: a linear chain, a frontier without merging, and Interference Search,
all with the same trained judge and the same cost unit (states judged)."""
import argparse
import json
import random
import time
from functools import lru_cache

from interference_search import linear_search, search
from interference_search.countdown import gen_hard_problem, gen_problem, moves
from interference_search.countdown_domain import Countdown
from interference_search.judge import load_judge

ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="50,150,200,500")
ap.add_argument("--six", type=int, default=0, help="also run this many hard 6-number problems (slow)")
ap.add_argument("--out", default="countdown_benchmark.json")
args = ap.parse_args()
BUDGETS = [int(b) for b in args.budgets.split(",")]


def solution_paths(p):
    t = p["target"]

    @lru_cache(maxsize=None)
    def f(s):
        return int(s[0] == t) if len(s) == 1 else sum(f(m) for m in moves(s))
    return f(tuple(sorted(p["numbers"])))


dom = Countdown(load_judge())
rng = random.Random(11)
sets = {"4 numbers, hard": [gen_hard_problem(rng) for _ in range(30)]}
if args.six:
    hard6, i = [], 0
    while len(hard6) < args.six:
        p = gen_problem(random.Random(60_000 + i), n_numbers=6)
        i += 1
        if solution_paths(p) <= 24:
            hard6.append(p)
    sets["6 numbers, <= 24 solution paths"] = hard6

out = {}
for name, probs in sets.items():
    print(f"\n{name} ({len(probs)} problems); solve rate at judged-state budget " + " ".join(f"{b:>6d}" for b in BUDGETS), flush=True)
    arms = {
        "one line of thought": lambda p, b: linear_search(dom, p, b, rng=random.Random(0)),
        "frontier, no merge": lambda p, b: search(dom, p, b, width=max(1, b // (3 * len(p["numbers"]))), merge=False),
        "Interference Search": lambda p, b: search(dom, p, b, width=max(1, b // (3 * len(p["numbers"]))), merge=True),
    }
    out[name] = {}
    for arm, fn in arms.items():
        rates, rounds, ms = [], 0.0, 0.0
        for b in BUDGETS:
            t = time.perf_counter()
            rs = [fn(p, b) for p in probs]
            ms = (time.perf_counter() - t) / len(probs) * 1000
            rates.append(sum(r.solved for r in rs) / len(probs))
            ok = [r.rounds for r in rs if r.solved]
            rounds = sum(ok) / max(len(ok), 1)
        out[name][arm] = {"solve": dict(zip(BUDGETS, rates)), "rounds_at_max_budget": rounds, "ms_per_problem_at_max_budget": ms}
        print(f"  {arm:22s}" + " ".join(f"{x:6.0%}" for x in rates) + f"   rounds {rounds:4.1f}   {ms:5.1f} ms/problem", flush=True)
json.dump(out, open(args.out, "w"), indent=1)
