"""End-to-end scheduler benchmark with timed model and tool stages.

The latencies are controlled async sleeps, not a measured LLM or real tool.
Both DAG arms have identical exact keys, model batch capacity, and execution
workers. The tree control uses the same concurrency without shared state.
"""

import argparse
import asyncio
import json
import random
import statistics

from interference_search.async_execution import AsyncAction, AsyncExecutionSearch


class TimedGraphDomain:
    def __init__(self, edges, delays, goal=None):
        self.edges = edges
        self.delays = delays
        self.goal = goal

    def start(self, problem):
        return "S" if self.goal is not None else (0, 0)

    def key(self, state):
        return state

    async def propose_batch(self, states, problem):
        await asyncio.sleep(0.003 + 0.001 * len(states))
        return [[AsyncAction(child) for child in self.edges.get(state, ())]
                for state in states]

    async def execute(self, state, action, problem):
        await asyncio.sleep(self.delays[(state, action)])
        return action

    def execution_key(self, state, action, problem):
        return (state, action)

    def is_goal(self, state, problem):
        return state == self.goal if self.goal is not None else False

    def certified_dead(self, state, problem):
        return False


def latency_problem(seed):
    rng = random.Random(seed)
    edges = {"S": ("A", "B"), "A": ("X",), "B": ("C",), "C": ("G",)}
    delays = {(parent, child): rng.uniform(0.002, 0.005)
              for parent, children in edges.items() for child in children}
    delays[("S", "A")] = rng.uniform(0.035, 0.055)
    return TimedGraphDomain(edges, delays, goal="G")


def throughput_problem(seed, depth=5):
    rng = random.Random(seed)
    edges = {}
    for level in range(depth):
        for index in range(level + 1):
            edges[(level, index)] = ((level + 1, index), (level + 1, index + 1))
    delays = {(parent, child): rng.uniform(0.002, 0.006)
              for parent, children in edges.items() for child in children}
    return TimedGraphDomain(edges, delays)


ARMS = {
    "pipeline_dag": dict(mode="pipeline", merge_states=True),
    "barrier_dag": dict(mode="barrier", merge_states=True),
    "pipeline_tree": dict(mode="pipeline", merge_states=False,
                          dedupe_executions=False),
}


async def run_all(seeds):
    rows = []
    for scenario, make_problem in (("goal_latency", latency_problem),
                                   ("exhaustive_throughput", throughput_problem)):
        for seed in range(seeds):
            domain = make_problem(seed)
            for arm, options in ARMS.items():
                scheduler = AsyncExecutionSearch(model_batch_size=8,
                                                 execution_workers=8, **options)
                result = await scheduler.run(domain, None, expansion_budget=128)
                rows.append({"scenario": scenario, "seed": seed, "arm": arm,
                             "solved": result.solved,
                             "wall_ms": round(result.wall_seconds * 1000, 3),
                             "expansions": result.expansions,
                             "proposal_batches": result.proposal_batches,
                             "execution_requests": result.execution_requests,
                             "execution_calls": result.execution_calls,
                             "state_merges": result.state_merges,
                             "peak_graph_nodes": result.peak_graph_nodes,
                             "cancelled_executions": result.cancelled_executions})
    groups = []
    for scenario in ("goal_latency", "exhaustive_throughput"):
        for arm in ARMS:
            subset = [row for row in rows if row["scenario"] == scenario
                      and row["arm"] == arm]
            groups.append({"scenario": scenario, "arm": arm,
                           "solved": sum(row["solved"] for row in subset),
                           "total": len(subset),
                           **{f"mean_{key}": round(statistics.mean(row[key] for row in subset), 3)
                              for key in ("wall_ms", "expansions", "proposal_batches",
                                          "execution_requests", "execution_calls",
                                          "state_merges", "peak_graph_nodes",
                                          "cancelled_executions")}})
    return {"config": {"seeds": seeds, "model_batch_size": 8,
                       "execution_workers": 8, "expansion_budget": 128,
                       "timing_source": "asyncio.sleep"},
            "groups": groups, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = asyncio.run(run_all(args.seeds))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({"config": result["config"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
