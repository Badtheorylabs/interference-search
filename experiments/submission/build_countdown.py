"""Freeze source-disjoint Countdown manifests with exact operation-path strata."""
import argparse
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

from interference_search.countdown import gen_problem, moves, reachable_states
from experiments.submission.common import identity, write_json


def path_count(problem):
    @lru_cache(None)
    def count(state):
        if len(state) == 1:
            return int(state[0] == problem["target"])
        return sum(count(child) for child in moves(state))
    return count(tuple(sorted(problem["numbers"])))


def stratum(count):
    return "1" if count == 1 else "2-5" if count <= 5 else "6-20" if count <= 20 else ">20"


def training_sources():
    # Recover the original 400 problems per size, generated with one seed-0 stream.
    rng = random.Random(0)
    return [gen_problem(rng, n_numbers=n) for n in (4, 5) for _ in range(400)]


def build(out, total=1000, seed=20261003, transfer=100, max_attempts=200000):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"manifest already exists: {out}")
    train = training_sources()
    forbidden = {identity(p) for p in train}
    # Exclude exact labelled training states as test roots as well as source problems.
    train_states = set()
    for p in train:
        train_states.update((s, p["target"]) for s in reachable_states(p["numbers"]))
    rng = random.Random(seed)
    quotas = {k: (total // 2) // 4 for k in ("1", "2-5", "6-20", ">20")}
    random_quota = total - sum(quotas.values())
    main, counts, attempts = [], Counter(), 0
    while len(main) < total and attempts < max_attempts:
        attempts += 1
        p = gen_problem(rng, n_numbers=4)
        key = identity(p)
        if key in forbidden or (tuple(sorted(p["numbers"])), p["target"]) in train_states:
            continue
        paths = path_count(p)
        bucket = stratum(paths)
        cohort = "random" if counts["random"] < random_quota else "stratified"
        if cohort == "stratified" and counts[bucket] >= quotas[bucket]:
            continue
        forbidden.add(key)
        p.update(problem_id=key, cohort=cohort, n_numbers=4, solution_paths=paths,
                 stratum=bucket)
        main.append(p)
        counts["random" if cohort == "random" else bucket] += 1
    if len(main) != total:
        raise RuntimeError(f"strata not filled after {attempts}: {dict(counts)}")
    generalisation = []
    # Freeze random large-size sets before looking at scores or thresholds.
    for n in (6, 7, 8):
        for _ in range(transfer):
            while True:
                p = gen_problem(rng, n_numbers=n)
                key = identity(p)
                if key not in forbidden:
                    break
            forbidden.add(key)
            p.update(problem_id=key, cohort="random", n_numbers=n)
            generalisation.append(p)
    write_json(out, {"version": 1, "seed": seed, "main": main,
                    "generalisation": generalisation,
                    "training_sources": [{**p, "problem_id": identity(p)} for p in train],
                    "main_counts": dict(counts), "generation_attempts": attempts,
                    "train_root_overlap": 0,
                    "path_definition": "multiplicity of legal operation sequences in moves(); commutative operations counted once per unordered index pair; equal values retain index multiplicity",
                    "last_bucket": ">20 (the review's final '20 paths' interpreted as more than 20)",
                    "split_policy": "source-problem disjoint plus main test roots absent from exact training-state/target pairs; substate overlap reported separately, never used to filter results"})
    print(f"frozen {len(main)} main and {len(generalisation)} transfer problems: {dict(counts)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--transfer", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    build(args.out, args.total, args.seed, args.transfer)


if __name__ == "__main__":
    main()
