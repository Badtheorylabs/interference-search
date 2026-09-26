"""Controlled counterexample search with matched candidate and verifier budgets.

Run: .venv/bin/python experiments/register_program_benchmark.py --seeds 10
Rounds are a latency-depth proxy only: the verifier here is a cheap local
function, so real wall time is also recorded. No learned model is involved.
"""

import argparse
from itertools import product
import json
import random
import statistics
import time

from interference_search.register_program import RegisterProgram
from interference_search.register_program_search import (
    CachedFactorSolver, CompactQuotientSolver, FullCandidateBlacklist,
)


POLICIES = {
    "blacklist": FullCandidateBlacklist,
    "cached_factor": CachedFactorSolver,
    "compact_bdd": CompactQuotientSolver,
}


def run_trial(task, policy_name, batch_size, call_budget, selection, seed):
    policy = POLICIES[policy_name](task.public_view())
    calls = rounds = repeated_register_feedback = 0
    solved = False
    seen_registers = set()
    grammar = task.public_view()
    proposed_candidates = signature_evals = 0
    preference_rng = random.Random(100_003 * seed + 7919)
    used_preferences = set()
    started = time.perf_counter()
    while calls < call_budget:
        requested = min(batch_size, call_budget - calls)
        pool_size = max(requested, 64) if selection == "coverage" else requested
        if selection in {"diverse", "coverage"}:
            pool_size = min(pool_size, (1 << task.decisions) - len(used_preferences))
            if pool_size == 0:
                break
        preferences = None
        if selection in {"diverse", "coverage"}:
            preferences = []
            while len(preferences) < pool_size:
                preferred = tuple(preference_rng.randrange(2)
                                  for _ in range(task.decisions))
                if preferred not in used_preferences:
                    used_preferences.add(preferred)
                    preferences.append(preferred)
        batch = policy.batch(pool_size, preferences)
        proposed_candidates += len(batch)
        if selection == "coverage" and len(batch) > requested:
            signatures = [tuple(grammar.local_output(register,
                          tuple(candidate[index] for index in grammar.scope(register)))
                          for register in range(grammar.registers)) for candidate in batch]
            signature_evals += len(batch) * grammar.registers
            chosen = []
            covered = [set() for _ in range(grammar.registers)]
            available = list(range(len(batch)))
            for _ in range(requested):
                best = max(available, key=lambda position: sum(
                    (1 << (grammar.registers - register - 1))
                    for register, value in enumerate(signatures[position])
                    if register not in seen_registers and value not in covered[register]))
                chosen.append(batch[best])
                for register, value in enumerate(signatures[best]):
                    covered[register].add(value)
                available.remove(best)
            batch = chosen
        if not batch:
            break
        assert len(batch) == len(set(batch))
        feedback = [task.verify(candidate) for candidate in batch]
        calls += len(batch)
        rounds += 1
        if None in feedback:
            solved = True
            break
        for item in feedback:
            if item.register in seen_registers:
                repeated_register_feedback += 1
            seen_registers.add(item.register)
            policy.update(item)
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "solved": solved,
        "calls": calls,
        "rounds": rounds,
        "repeat_feedback": repeated_register_feedback,
        "certificate_patterns": policy.stats.certificate_patterns,
        "forbidden_cubes_applied": policy.stats.forbidden_cubes_applied,
        "selection_nodes": policy.stats.selection_nodes,
        "proposed_candidates": proposed_candidates,
        "signature_evals": signature_evals,
        "peak_compact_nodes": policy.stats.peak_compact_nodes,
        "represented_after_last_update": policy.stats.represented_after_last_update,
        "wall_ms": elapsed_ms,
    }
    return result


def benchmark(seeds, call_budget, registers, slots, topologies, batch_sizes, selections):
    rows = []
    output_diversity = {}
    for topology in topologies:
        diversity_values = []
        for seed in range(seeds):
            task = RegisterProgram(registers=registers, slots_per_register=slots,
                                   topology=topology, seed=seed)
            grammar = task.public_view()
            for register in range(registers):
                diversity_values.append(len({grammar.local_output(register, pattern)
                    for pattern in product((0, 1), repeat=len(grammar.scope(register)))}))
            for batch_size in batch_sizes:
                for selection in selections:
                    for policy_name in POLICIES:
                        result = run_trial(task, policy_name, batch_size,
                                           call_budget, selection, seed)
                        rows.append({"topology": topology, "seed": seed,
                                     "batch_size": batch_size, "selection": selection,
                                     "policy": policy_name, **result})
        output_diversity[topology] = {
            "mean": round(statistics.mean(diversity_values), 3),
            "min": min(diversity_values), "max": max(diversity_values),
        }
    groups = []
    for topology in topologies:
        for batch_size in batch_sizes:
            for selection in selections:
                for policy_name in POLICIES:
                    subset = [row for row in rows if row["topology"] == topology
                              and row["batch_size"] == batch_size
                              and row["selection"] == selection
                              and row["policy"] == policy_name]
                    groups.append({
                        "topology": topology, "batch_size": batch_size,
                        "selection": selection,
                        "policy": policy_name,
                        "solved": sum(row["solved"] for row in subset),
                        "total": len(subset),
                        **{f"mean_{key}": round(statistics.mean(row[key] for row in subset), 3)
                           for key in ("calls", "rounds", "repeat_feedback",
                                       "certificate_patterns", "forbidden_cubes_applied",
                                       "selection_nodes", "proposed_candidates",
                                       "signature_evals", "peak_compact_nodes", "wall_ms")},
                    })
    return {"config": {"seeds": seeds, "call_budget": call_budget,
                       "registers": registers, "slots_per_register": slots,
                       "topologies": topologies, "batch_sizes": batch_sizes,
                       "selections": selections},
            "output_diversity": output_diversity, "groups": groups, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--call-budget", type=int, default=128)
    parser.add_argument("--registers", type=int, default=6)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--topologies", nargs="+",
                        default=["independent", "ring", "sparse"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--selections", nargs="+",
                        default=["lexicographic", "diverse", "coverage"],
                        choices=["lexicographic", "diverse", "coverage"])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = benchmark(args.seeds, args.call_budget, args.registers, args.slots,
                       args.topologies, args.batch_sizes, args.selections)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({"config": result["config"],
                      "output_diversity": result["output_diversity"],
                      "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
