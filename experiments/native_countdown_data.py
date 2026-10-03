"""Build a frozen exact-label curriculum for the native frontier mechanism.

Each row contains several merged first-level states and one verified dead
second-level state reachable from at least two parents. The refutation gives
an exact cross-branch target without giving the model a solution at inference.
This is a Countdown mechanism curriculum, not a general reasoning eval.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from interference_search.countdown import children, gen_problem, solver
from interference_search.countdown_actions import CountdownActionCodec


CODEC = CountdownActionCodec(max_numbers=7)


def make_row(seed: int, n_numbers: int, width: int = 8):
    problem = gen_problem(random.Random(seed), n_numbers=n_numbers)
    start = tuple(sorted(problem["numbers"]))
    target = problem["target"]
    can_solve = solver(target)
    candidates = children(start)
    if not any(can_solve(state) for state in candidates):
        return None
    dead_successors: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for state in candidates:
        for action_id, legal in enumerate(CODEC.legal_mask(state)):
            if not legal:
                continue
            successor = CODEC.apply(state, action_id)
            if not can_solve(successor):
                dead_successors[successor].add(state)
    shared = [state for state, parents in dead_successors.items() if len(parents) >= 2]
    if not shared:
        return None
    rng = random.Random(seed + 700_000)
    refuted = rng.choice(sorted(shared))
    selected = list(sorted(dead_successors[refuted]))[:2]
    alive = [state for state in candidates if can_solve(state) and state not in selected]
    if alive and len(selected) < width:
        selected.append(rng.choice(alive))
    remaining = [state for state in candidates if state not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, width - len(selected))])
    selected = selected[:width]
    if len(selected) < 2:
        return None

    action_targets = []
    sampled_successors = []
    refutation_actions = []
    for state in selected:
        legal = [k for k, allowed in enumerate(CODEC.legal_mask(state)) if allowed]
        winning = [k for k in legal if can_solve(CODEC.apply(state, k))]
        action_targets.append(winning[0] if winning else -100)
        refuted_actions = [k for k in legal if CODEC.apply(state, k) == refuted]
        refutation_actions.append(refuted_actions)
        samples = list(dict.fromkeys((winning[:1] + refuted_actions[:1] + rng.sample(legal, min(2, len(legal))))))
        sampled_successors.append([[k, list(CODEC.apply(state, k))] for k in samples])
    assert sum(bool(actions) for actions in refutation_actions) >= 2
    return {
        "seed": seed,
        "numbers": problem["numbers"],
        "target": target,
        "frontier": [list(state) for state in selected],
        "verified_dead_state": list(refuted),
        "survival_targets": [int(can_solve(state)) for state in selected],
        "action_targets": action_targets,
        "refutation_actions": refutation_actions,
        "sampled_successors": sampled_successors,
    }


def build_split(count: int, seed_start: int, n_numbers: int, excluded: set[tuple], max_attempts: int):
    rows = []
    attempted = 0
    while len(rows) < count and attempted < max_attempts:
        seed = seed_start + attempted
        attempted += 1
        row = make_row(seed, n_numbers)
        if row is None:
            continue
        identity = (tuple(sorted(row["numbers"])), row["target"])
        if identity in excluded:
            continue
        excluded.add(identity)
        rows.append(row)
    if len(rows) != count:
        raise RuntimeError(f"only built {len(rows)} of {count} rows after {attempted} attempts")
    return rows, attempted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/native_countdown"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = (("train", args.train, 0, 4),
             ("validation", args.validation, 1_000_000, 4),
             ("test_5_numbers", args.test, 2_000_000, 5))
    excluded: set[tuple] = set()
    report = {"generator": "native_countdown_data.py",
              "split_rule": "nonoverlapping seed ranges and exact problem identity dedupe; shared generator",
              "claim_boundary": "instance-disjoint Countdown curriculum; five-number structural holdout; no source-disjoint domain transfer",
              "splits": {}}
    for name, count, seed_start, n_numbers in specs:
        rows, attempted = build_split(count, seed_start, n_numbers, excluded, max_attempts=count * 50)
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        report["splits"][name] = {"rows": len(rows), "attempts": attempted, "numbers": n_numbers,
                                   "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "cross_branch_refutation_rows": len(rows)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
