"""Record real search traces for the animation. Countdown only (deterministic, CPU).

For each showcase problem: every round of Interference Search (all children the environment produced,
how many were duplicates merged away, the judge's score, which survived, and exact ground truth
alive/dead from DP), plus the linear chain's step-by-step path with the same judge. Also solve-rate
curves over budget for the 30 hard problems, and the code race from run logs when they exist."""
import json
import random
import re
from functools import lru_cache

from interference_search import linear_search, search
from interference_search.countdown import gen_hard_problem, moves
from interference_search.countdown_domain import Countdown
from interference_search.judge import load_judge

judge = load_judge()
dom = Countdown(judge)
rng = random.Random(11)
problems = [gen_hard_problem(rng) for _ in range(30)]


def truth(p):
    t = p["target"]

    @lru_cache(maxsize=None)
    def f(s):
        return s[0] == t if len(s) == 1 else any(f(m) for m in moves(s))
    return f


def frontier_trace(p, width=12):
    alive = truth(p)
    live, seen, rounds = [tuple(sorted(p["numbers"]))], set(), []
    for r in range(10):
        raw_children, parent_of = [], {}
        for s in live:
            seen.add(s)
            for c in moves(s):
                raw_children.append(c)
                parent_of.setdefault(c, s)
        goal = [c for c in raw_children if c == (p["target"],)]
        uniq = sorted({c for c in raw_children if len(c) > 1 and c not in seen})
        sc = judge(uniq, p["target"]) if uniq else []
        order = sorted(range(len(uniq)), key=lambda i: -sc[i])
        keep = set(order[:width])
        rounds.append({
            "round": r + 1,
            "live": [list(s) for s in live],
            "raw": len(raw_children),
            "merged_away": len(raw_children) - len(set(raw_children)),
            "candidates": [{"s": list(uniq[i]), "score": round(sc[i], 4), "alive": alive(uniq[i]),
                            "kept": i in keep, "parent": list(parent_of[uniq[i]])} for i in range(len(uniq))],
            "goal": bool(goal),
        })
        if goal:
            break
        live = [uniq[i] for i in order[:width]]
        if not live:
            break
    return rounds


def linear_trace(p, max_steps=120, temp=0.5):
    import math
    alive, rr = truth(p), random.Random(0)
    steps, s, start = [], tuple(sorted(p["numbers"])), tuple(sorted(p["numbers"]))
    for k in range(max_steps):
        kids = sorted(set(moves(s)))
        if (p["target"],) in kids:
            steps.append({"s": list(s), "event": "solved"})
            return steps
        kids = [c for c in kids if len(c) > 1]
        if not kids:
            steps.append({"s": list(s), "event": "dead end, restart", "alive": alive(s)})
            s = start
            continue
        sc = judge(kids, p["target"])
        w = [math.exp(math.log(max(x, 1e-6)) / temp) for x in sc]
        nxt = rr.choices(kids, weights=w)[0]
        steps.append({"s": list(s), "event": "step", "alive": alive(s)})
        s = nxt
    return steps


# pick showcase problems: the linear chain needs many steps, the frontier solves
show = []
for i, p in enumerate(problems):
    lt = linear_trace(p)
    ft = frontier_trace(p)
    show.append((len(lt), i, p, lt, ft))
show.sort(key=lambda x: -x[0])
showcase = [{"idx": i, "numbers": p["numbers"], "target": p["target"], "solution": p["solution"],
             "frontier": ft, "linear": lt} for _, i, p, lt, ft in show if ft[-1]["goal"]][:4]

# solve-rate curves on the 30 hard problems, same judge, cost = judged states
budgets = [10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 260, 330, 420, 520]
curves = {"linear chain": [], "Interference Search": []}
for b in budgets:
    curves["linear chain"].append(sum(linear_search(dom, p, b, rng=random.Random(0)).solved for p in problems))
    curves["Interference Search"].append(sum(search(dom, p, b, width=max(1, b // 12)).solved for p in problems))

# code race, from the real per-round logs
race = {}
for fname, arm in (("bench_code.log", "best-of-N"), ("bench_code_run3.log", "transcript"),
                   ("bench_code_run5.log", "linear state"), ("bench_code_run5.log", "Interference Search")):
    pts = []
    import os
    if not os.path.exists(fname):
        continue                                   # run logs are not shipped; the race data lives in results/llm/viz_slim.json
    for line in open(fname):
        m = re.match(rf"\s+\[{re.escape(arm)}\] round (\d+): .*solved so far (\d+)/30, mean tokens (\d+)", line)
        if m:
            pts.append({"round": int(m.group(1)), "solved": int(m.group(2)), "tokens": int(m.group(3))})
    race[arm] = pts

json.dump({"showcase": showcase, "budgets": budgets, "curves": curves, "cot_baseline": 3, "code_race": race},
          open("viz_data.json", "w"))
print("showcase:", [(s["idx"], s["numbers"], s["target"], len(s["linear"]), len(s["frontier"])) for s in showcase])
print("curves:", curves)
print("race points:", {k: len(v) for k, v in race.items()})
