"""Compare the shared reasoning graph with the released Countdown frontier.

All arms use the same frozen judge and expansion budgets. This is an
exploratory comparison of search schedulers, not a trained-model result.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from interference_search.countdown import gen_problem
from interference_search.countdown_parallel import CountdownParallelDomain
from interference_search.judge import load_judge
from interference_search.parallel_reasoning import ParallelReasoner
from searches import frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n6", type=int, default=100)
    parser.add_argument("--n7", type=int, default=30)
    parser.add_argument("--budgets", default="20,50,100,200,400")
    parser.add_argument("--out", type=Path, default=Path("results/countdown/parallel_reasoning_v2.json"))
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    if min([args.n6, args.n7] + budgets) < 1:
        parser.error("counts and budgets must be positive")
    judge = load_judge()
    domain = CountdownParallelDomain(judge)
    arm_names = ("released frontier", "shared graph", "shared graph + reserve",
                 "shared graph + reserve + mass")
    report = {
        "status": "exploratory same-judge scheduler comparison; no model-native claim",
        "budgets": budgets,
        "arms": arm_names,
        "problem_sets": {},
    }
    for numbers, count, seed_start in ((6, args.n6, 40_000), (7, args.n7, 50_000)):
        problems = [gen_problem(random.Random(seed_start + index), n_numbers=numbers)
                    for index in range(count)]
        by_arm = {}
        for name in arm_names:
            by_budget = {}
            for budget in budgets:
                rows = []
                for problem in problems:
                    started = time.perf_counter()
                    if name == "released frontier":
                        solved, expansions, rounds = frontier(
                            judge, problem["numbers"], problem["target"], budget
                        )
                        row = {"solved": solved, "expansions": expansions, "rounds": rounds}
                    else:
                        reasoner = ParallelReasoner(
                            width=max(1, budget // (numbers - 1)),
                            reserve="reserve" in name,
                            mass_weight=0.3 if "mass" in name else 0.0,
                        )
                        result = reasoner.run(domain, problem, expansion_budget=budget)
                        row = {
                            "solved": result.solved, "expansions": result.expansions,
                            "rounds": result.rounds, "judged_states": result.judged_states,
                            "proposal_count": result.proposal_count,
                            "merged_proposals": result.merged_proposals,
                            "certified_dead_nodes": result.certified_dead_nodes,
                            "rescue_rounds": result.rescue_rounds,
                            "peak_graph_nodes": result.peak_graph_nodes,
                            "peak_reserve": result.peak_reserve,
                        }
                    row["wall_ms"] = (time.perf_counter() - started) * 1000
                    assert row["expansions"] <= budget
                    rows.append(row)
                solved_rows = [row for row in rows if row["solved"]]
                by_budget[str(budget)] = {
                    "solved": len(solved_rows), "of": count,
                    "mean_expansions": statistics.mean(row["expansions"] for row in rows),
                    "mean_rounds_on_solved": statistics.mean(row["rounds"] for row in solved_rows)
                    if solved_rows else None,
                    "mean_wall_ms": statistics.mean(row["wall_ms"] for row in rows),
                    "mean_judged_states": statistics.mean(row.get("judged_states", 0) for row in rows)
                    if name != "released frontier" else None,
                    "mean_rescue_rounds": statistics.mean(row.get("rescue_rounds", 0) for row in rows)
                    if name != "released frontier" else None,
                    "mean_peak_graph_nodes": statistics.mean(row.get("peak_graph_nodes", 0) for row in rows)
                    if name != "released frontier" else None,
                }
                print(f"{numbers} numbers | {name} | budget {budget}: {len(solved_rows)}/{count}", flush=True)
            by_arm[name] = by_budget
        report["problem_sets"][str(numbers)] = {"count": count, "seed_start": seed_start,
                                                  "arms": by_arm}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "status": report["status"]}))


if __name__ == "__main__":
    main()
