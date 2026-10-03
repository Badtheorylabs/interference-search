"""Paired analysis of frozen MBPP pipeline/barrier receipts."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def analyze(receipt, bootstrap_seed=112, bootstrap_samples=20_000):
    if receipt["status"] != "complete":
        raise ValueError("incomplete benchmark receipt")
    pairs = []
    for task_id in receipt["task_ids"]:
        rows = {row["mode"]: row for row in receipt["rows"]
                if row["task_id"] == task_id}
        if set(rows) != {"pipeline", "barrier"}:
            raise ValueError(f"task {task_id} is not paired")
        pairs.append((rows["pipeline"], rows["barrier"]))
    savings = [barrier["wall_seconds"] - pipeline["wall_seconds"]
               for pipeline, barrier in pairs]
    rng = random.Random(bootstrap_seed)
    resamples = sorted(
        statistics.mean(rng.choices(savings, k=len(savings)))
        for _ in range(bootstrap_samples)
    )
    groups = {}
    for mode in ("pipeline", "barrier"):
        rows = [pair[0 if mode == "pipeline" else 1] for pair in pairs]
        times = sorted(row["wall_seconds"] for row in rows)
        groups[mode] = {
            "solved": sum(row["solved"] for row in rows),
            "mean_wall_seconds": round(statistics.mean(times), 5),
            "median_wall_seconds": round(statistics.median(times), 5),
            "p95_wall_seconds": round(times[math.ceil(.95 * len(times)) - 1], 5),
            **{f"total_{key}": sum(row[key] for row in rows)
               for key in ("model_requests", "completed_model_requests",
                           "cancelled_model_requests", "started_prompt_tokens",
                           "output_tokens", "execution_calls", "state_merges",
                           "execution_coalesced", "sandbox_errors")},
        }
    return {
        "tasks": len(pairs), "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "pipeline_faster_pairs": sum(value > 0 for value in savings),
        "equal_pairs": sum(value == 0 for value in savings),
        "mean_seconds_saved": round(statistics.mean(savings), 5),
        "median_seconds_saved": round(statistics.median(savings), 5),
        "mean_saving_95pct_bootstrap": [
            round(resamples[int(.025 * bootstrap_samples)], 5),
            round(resamples[int(.975 * bootstrap_samples)], 5)],
        "median_paired_speed_ratio": round(statistics.median(
            barrier["wall_seconds"] / pipeline["wall_seconds"]
            for pipeline, barrier in pairs), 5),
        "paired_solve_mismatches": sum(pipeline["solved"] != barrier["solved"]
                                       for pipeline, barrier in pairs),
        "same_prompt_output_mismatches_total": receipt[
            "same_prompt_output_mismatches_total"],
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(json.loads(Path(args.receipt).read_text()))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
