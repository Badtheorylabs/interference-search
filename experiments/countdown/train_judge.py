"""Train the Countdown judge on exact labels, then test it by pruned merged-state search.

Training data: every reachable state of random 4- and 5-number problems, labelled alive or dead by exact
DP. Test: breadth-first search over merged states that keeps only states the judge calls alive, on unseen
6-number problems (random and hard), and on 7-number problems, against no pruning, a sound hand bound and
the exact oracle.
"""
import argparse
import json
import random
import time
from functools import lru_cache

import torch
import torch.nn as nn

from interference_search.countdown import gen_problem, moves, reachable_states, solver
from interference_search.judge import Refuter, encode, hand_alive

ap = argparse.ArgumentParser()
ap.add_argument("--train-inst", type=int, default=400, help="instances per size (4 and 5 numbers)")
ap.add_argument("--test-inst", type=int, default=100)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hard-cap", type=int, default=24, help="max solution paths for the hard test set")
ap.add_argument("--out", default="countdown_judge_retrained.pt")
args = ap.parse_args()
torch.manual_seed(args.seed)
torch.set_num_threads(2)
t0 = time.time()

rng, rows = random.Random(args.seed), []
for n in (4, 5):
    for _ in range(args.train_inst):
        p = gen_problem(rng, n_numbers=n)
        solvable = solver(p["target"])
        rows += [(s, p["target"], solvable(s)) for s in reachable_states(p["numbers"])]
random.Random(1).shuffle(rows)
n_alive = sum(r[2] for r in rows)
print(f"train states {len(rows)}  alive {n_alive / len(rows):.1%}  ({time.time() - t0:.0f}s)", flush=True)

X, M = encode([r[0] for r in rows], [r[1] for r in rows])
Y = torch.tensor([float(r[2]) for r in rows])
model = Refuter()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
pos_w = torch.tensor((len(rows) - n_alive) / max(n_alive, 1))
for ep in range(args.epochs):
    tot = 0.0
    for i in range(0, len(rows), 512):
        loss = nn.functional.binary_cross_entropy_with_logits(model(X[i:i + 512], M[i:i + 512]), Y[i:i + 512], pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(Y[i:i + 512])
    print(f"epoch {ep} loss {tot / len(rows):.4f} ({time.time() - t0:.0f}s)", flush=True)
model.eval()
torch.save(model.state_dict(), args.out)


def solution_paths(p):
    t = p["target"]

    @lru_cache(maxsize=None)
    def f(s):
        return int(s[0] == t) if len(s) == 1 else sum(f(m) for m in moves(s))
    return f(tuple(sorted(p["numbers"])))


def make_test(n_numbers, count, seed0, hard):
    out, i = [], 0
    while len(out) < count:
        p = gen_problem(random.Random(seed0 + i), n_numbers=n_numbers)
        i += 1
        if not hard or solution_paths(p) <= args.hard_cap:
            out.append(p)
    return out


def pruned_bfs(numbers, target, keep):
    frontier, expanded = {tuple(sorted(numbers))}, 0
    while frontier:
        nxt = set()
        for s in frontier:
            expanded += 1
            nxt.update(moves(s))
        if any(len(s) == 1 for s in nxt):
            return (target,) in nxt, expanded
        cand = sorted(nxt)
        frontier = {s for s, k in zip(cand, keep(cand, target)) if k}
    return False, expanded


def learned(tau):
    def keep(states, target):
        with torch.no_grad():
            Xs, Ms = encode(states, target)
            return (torch.sigmoid(model(Xs, Ms)) >= tau).tolist()
    return keep


def suite(name, test, oracle=True):
    methods = {"no pruning": lambda st, t: [True] * len(st), "hand bound": lambda st, t: [hand_alive(s, t) for s in st]}
    for tau in (0.2, 0.5, 0.7, 0.9, 0.95, 0.99):
        methods[f"learned tau={tau}"] = learned(tau)
    res = {}
    if oracle:
        solved = exp = 0
        for p in test:
            solv = solver(p["target"])
            ok, e = pruned_bfs(p["numbers"], p["target"], lambda st, t: [solv(s) for s in st])
            solved += ok; exp += e
        res["exact oracle"] = {"solved": solved / len(test), "expanded": exp / len(test)}
    for m, keep in methods.items():
        solved = exp = 0
        for p in test:
            ok, e = pruned_bfs(p["numbers"], p["target"], keep)
            solved += ok; exp += e
        res[m] = {"solved": solved / len(test), "expanded": exp / len(test)}
    base = res["no pruning"]["expanded"]
    print(f"\n{name} ({len(test)} problems)\nmethod              solved   states expanded   vs no pruning")
    for m, r in res.items():
        print(f"{m:18s}  {r['solved']:6.1%}   {r['expanded']:12.0f}      {r['expanded'] / base:6.1%}", flush=True)
    return res


out = {"6 numbers, random": suite("unseen 6-number problems, random", make_test(6, args.test_inst, 10_000, False)),
       "6 numbers, hard": suite(f"unseen 6-number problems, <= {args.hard_cap} solution paths", make_test(6, args.test_inst, 20_000, True)),
       "7 numbers, random": suite("unseen 7-number problems", make_test(7, 10, 30_000, False), oracle=False)}
json.dump(out, open(args.out.replace(".pt", "_results.json"), "w"), indent=1)
