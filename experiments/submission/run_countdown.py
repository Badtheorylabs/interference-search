"""One raw row per problem x method x seed x budget, with resumable receipts."""
import argparse
import json
import time
from pathlib import Path

import torch

from interference_search.countdown import reachable_states
from interference_search.judge import load_judge
from experiments.submission.common import ROOT, append, digest, environment, read_rows, write_json
from experiments.submission.countdown_engine import methods, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights/countdown_judge.pt"))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--budgets", default="25,50,100,200,500,1500")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget-unit", choices=("state_work", "proposed"), default="state_work")
    ap.add_argument("--uncached", action="store_true", help="end-to-end timing, no cross-trial score replay")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = json.loads(Path(args.manifest).read_text())["main"][args.offset:]
    if args.limit:
        problems = problems[:args.limit]
    seeds = [int(v) for v in args.seeds.split(",")]
    budgets = [int(v) for v in args.budgets.split(",")]
    arms = methods()
    if args.methods != "all":
        wanted = args.methods.split(",")
        arms = [m for m in arms if m.name in wanted]
        if set(wanted) != {m.name for m in arms}:
            raise ValueError("unknown methods")
    config = {"manifest_sha256": digest(args.manifest), "weights_sha256": digest(args.weights),
              "seeds": seeds, "budgets": budgets, "methods": [m.__dict__ for m in arms],
              "width": args.width, "threshold": args.threshold, "budget_unit": args.budget_unit,
              "problem_ids": [p["problem_id"] for p in problems], "uncached": args.uncached,
              "source_sha256": digest(Path(__file__).with_name("countdown_engine.py"))}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory belongs to a different immutable configuration")
    write_json(config_path, config)
    if not (out / "environment.json").exists():
        write_json(out / "environment.json", environment())
    done = {(r["problem_id"], r["method"], r["seed"], r["budget"]) for r in read_rows(out / "raw.jsonl")}
    judge = load_judge(args.weights)
    t0 = time.perf_counter()
    raw_path = out / "raw.jsonl"
    for index, problem in enumerate(problems):
        missing = [(m, s, b) for m in arms for s in seeds for b in budgets
                   if (problem["problem_id"], m.name, s, b) not in done]
        if not missing:
            continue
        cache = {}
        replay_seconds = 0.0
        if not args.uncached:
            started = time.perf_counter()
            states = sorted(reachable_states(problem["numbers"]))
            for begin in range(0, len(states), 1024):
                batch = states[begin:begin + 1024]
                cache.update(zip(batch, judge(batch, problem["target"])))
            replay_seconds = time.perf_counter() - started
            append(out / "score_precompute.jsonl", {"problem_id": problem["problem_id"],
                   "states_scored": len(cache), "wall_seconds": replay_seconds,
                   "scores": [[list(state), value] for state, value in sorted(cache.items())]})
        def score(states, target):
            return judge(states, target) if args.uncached else [cache[state] for state in states]
        for method, seed, budget in missing:
            row = run(problem, method, seed, budget, score, args.width, args.threshold, args.budget_unit)
            row.update(problem_id=problem["problem_id"], method=method.name, seed=seed,
                       budget=budget, cohort=problem["cohort"], stratum=problem["stratum"],
                       solution_paths=problem["solution_paths"], n_numbers=problem["n_numbers"],
                       score_mode="online_uncached" if args.uncached else "frozen_judge_replay",
                       score_precompute_seconds=replay_seconds,
                       model_hash=config["weights_sha256"])
            append(raw_path, row)
        print(f"problem {index + 1}/{len(problems)}; {len(missing)} trials; elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    expected = len(problems) * len(arms) * len(seeds) * len(budgets)
    write_json(out / "complete.json", {"expected_rows": expected,
               "actual_rows": sum(1 for _ in raw_path.open()), "elapsed_seconds_this_invocation": time.perf_counter()-t0})


if __name__ == "__main__":
    main()
