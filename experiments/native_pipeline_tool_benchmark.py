"""H100 model-work and OS-tool overlap with a fixed action graph.

The 4B backbone and frontier head run for every proposal batch. Their output
is intentionally discarded so both schedulers execute the same graph. Tools
are bounded, cancellable /bin/sleep subprocesses with variable latency.
This isolates scheduler overlap under real GPU forwards and process execution.
"""

import argparse
import asyncio
import json
import random
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.async_execution import AsyncAction, AsyncExecutionSearch
from interference_search.native_async_countdown import NativeAsyncCountdownDomain
from interference_search.native_lm import NativeFrontierLM


class FixedGraphNativeDomain:
    def __init__(self, model, tokenizer, seed):
        self.native = NativeAsyncCountdownDomain(model, tokenizer, max_numbers=4,
                                                actions_per_state=2, state_mode="auto")
        self.graph = {
            (0, 0): ((1, 0), (1, 1)),
            (1, 0): ((2, 0),),
            (1, 1): ((2, 1),),
            (2, 1): ((3, 1),),
        }
        rng = random.Random(seed)
        self.delays = {(parent, child): rng.uniform(0.008, 0.014)
                       for parent, children in self.graph.items() for child in children}
        self.delays[((0, 0), (1, 0))] = rng.uniform(0.10, 0.14)

    def start(self, problem):
        return (0, 0)

    def key(self, state):
        return state

    async def prepare(self, problem):
        return await self.native.prepare(problem)

    async def propose_batch(self, states, context):
        await self.native.propose_batch(states, context)
        return [[AsyncAction(child, priority=1.0 if child[1] == 1 else 0.0)
                 for child in self.graph.get(state, ())] for state in states]

    async def execute(self, state, action, context):
        process = await asyncio.create_subprocess_exec(
            "/bin/sleep", f"{self.delays[(state, action)]:.4f}")
        try:
            code = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
        if code:
            raise RuntimeError("tool subprocess failed")
        return action

    def execution_key(self, state, action, context):
        return (state, action)

    def is_goal(self, state, problem):
        return state == (3, 1)

    def certified_dead(self, state, problem):
        return False


async def run_case(model, tokenizer, seed, arm):
    domain = FixedGraphNativeDomain(model, tokenizer, seed)
    scheduler = AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                     mode=arm, merge_states=True)
    problem = {"numbers": [2, 3, 4, 5], "target": 19,
               "task_text": "Numbers: [2, 3, 4, 5]. Target: 19. Search exact paths. " * 8}
    result = await scheduler.run(domain, problem, expansion_budget=10)
    torch.cuda.synchronize()
    return {"seed": seed, "arm": arm, "solved": result.solved,
            "wall_ms": round(result.wall_seconds * 1000, 3),
            "expansions": result.expansions,
            "model_batches": result.proposal_batches,
            "tool_calls": result.execution_calls,
            "cancelled_tools": result.cancelled_executions,
            "encoder_routes": domain.native.encoder_routes}


async def measure(model, tokenizer, seeds):
    warm = FixedGraphNativeDomain(model, tokenizer, 0)
    problem = {"numbers": [2, 3, 4, 5], "target": 19,
               "task_text": "Numbers: [2, 3, 4, 5]. Target: 19. Search exact paths. " * 8}
    prefix_tokens = len(tokenizer.encode(problem["task_text"], add_special_tokens=False))
    context = await warm.prepare(problem)
    await warm.native.propose_batch([(0, 0)], context)
    torch.cuda.synchronize()
    rows = []
    for seed in range(seeds):
        order = ("pipeline", "barrier") if seed % 2 == 0 else ("barrier", "pipeline")
        for arm in order:
            rows.append(await run_case(model, tokenizer, seed, arm))
    groups = []
    for arm in ("pipeline", "barrier"):
        subset = [row for row in rows if row["arm"] == arm]
        groups.append({"arm": arm, "solved": sum(row["solved"] for row in subset),
                       "total": len(subset),
                       "median_wall_ms": round(statistics.median(
                           row["wall_ms"] for row in subset), 3),
                       "mean_wall_ms": round(statistics.mean(
                           row["wall_ms"] for row in subset), 3),
                       "mean_model_batches": round(statistics.mean(
                           row["model_batches"] for row in subset), 3),
                       "mean_tool_calls": round(statistics.mean(
                           row["tool_calls"] for row in subset), 3)})
    return {"config": {"seeds": seeds, "model_batch_size": 2,
                       "execution_workers": 2, "tool": "/bin/sleep subprocess",
                       "prefix_tokens": prefix_tokens,
                       "action_policy": "fixed graph; model forward output discarded"},
            "groups": groups, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    torch.manual_seed(11)
    model = NativeFrontierLM(backbone, actions=48, heads=4).to(
        device="cuda", dtype=torch.bfloat16).eval()
    result = asyncio.run(measure(model, tokenizer, args.seeds))
    result.update({"model": args.model, "device": torch.cuda.get_device_name(0),
                   "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2)})
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({"config": result["config"], "groups": result["groups"],
                      "model": result["model"], "peak_cuda_mib": result["peak_cuda_mib"]},
                     indent=2))


if __name__ == "__main__":
    main()
