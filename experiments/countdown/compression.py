"""How much path compression does Countdown have? Exact dynamic programming.

For each depth: distinct operation sequences (what a path-by-path reasoner walks), distinct states (what
a merged-state reasoner holds), how many states are already dead, and the share of all full paths that
run through a dead state, which is the work one state-level refutation removes without enumerating it.
"""
import random
from collections import Counter, defaultdict
from functools import lru_cache

from interference_search.countdown import gen_problem, moves, solver


def analyze(numbers, target):
    solvable = solver(target)

    @lru_cache(maxsize=None)
    def completions(state):
        return 1 if len(state) == 1 else sum(completions(n) for n in moves(state))

    rows, frontier, depth = [], Counter({tuple(sorted(numbers)): 1}), 0
    while True:
        dead = [s for s in frontier if not solvable(s)]
        rows.append({"depth": depth, "paths": sum(frontier.values()), "states": len(frontier),
                     "dead_states": len(dead),
                     "full_paths_through_dead": sum(frontier[s] * completions(s) for s in dead),
                     "full_paths": sum(frontier[s] * completions(s) for s in frontier)})
        if all(len(s) == 1 for s in frontier):
            return rows
        nxt = Counter()
        for s, c in frontier.items():
            for n in moves(s):
                nxt[n] += c
        frontier, depth = nxt, depth + 1


if __name__ == "__main__":
    rng = random.Random(0)
    for n_numbers in (4, 5, 6):
        agg = defaultdict(lambda: defaultdict(float))
        for _ in range(20):
            p = gen_problem(rng, n_numbers=n_numbers)
            for r in analyze(p["numbers"], p["target"]):
                for k, v in r.items():
                    agg[r["depth"]][k] += v / 20
        print(f"\n{n_numbers} numbers, 20 random solvable instances (mean per instance)")
        print("depth      paths      states   paths/state   dead states   share of full paths through dead states")
        for d in sorted(agg):
            a = agg[d]
            print(f"{d:5d} {a['paths']:11.0f} {a['states']:11.0f}   {a['paths'] / max(a['states'], 1):10.1f}   "
                  f"{a['dead_states']:10.0f} ({a['dead_states'] / max(a['states'], 1):5.1%})   "
                  f"{a['full_paths_through_dead'] / max(a['full_paths'], 1):6.1%}")
