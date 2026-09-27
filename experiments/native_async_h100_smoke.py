"""Bounded native-weight inference and async-runtime smoke on one H100.

Uses a cached Qwen3-4B-Base checkpoint and a freshly initialized frontier
head. It measures engineering feasibility only; the head is not trained.
"""

import argparse
import asyncio
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.async_execution import AsyncExecutionSearch
from interference_search.native_async_countdown import NativeAsyncCountdownDomain
from interference_search.native_lm import NativeFrontierLM


async def run_arm(model, tokenizer, mode, budget):
    problem = {"numbers": [24, 98, 19, 3], "target": 361}
    domain = NativeAsyncCountdownDomain(model, tokenizer, max_numbers=4,
                                        actions_per_state=4, coupling="zero")
    scheduler = AsyncExecutionSearch(model_batch_size=8, execution_workers=8,
                                     mode=mode)
    result = await scheduler.run(domain, problem, expansion_budget=budget)
    return {"mode": mode, "solved": result.solved,
            "wall_seconds": round(result.wall_seconds, 4),
            "expansions": result.expansions,
            "proposal_batches": result.proposal_batches,
            "proposed_actions": result.proposed_actions,
            "execution_calls": result.execution_calls,
            "state_merges": result.state_merges,
            "peak_graph_nodes": result.peak_graph_nodes,
            "prepare_calls": result.prepare_calls,
            "observations": result.observations,
            "predictions_recorded": result.predictions_recorded,
            "encoder_routes": domain.encoder_routes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(11)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = NativeFrontierLM(backbone, actions=48, heads=4).to(
        device="cuda", dtype=torch.bfloat16).eval()
    load_seconds = time.perf_counter() - started
    warmup_started = time.perf_counter()
    asyncio.run(run_arm(model, tokenizer, "pipeline", 2))
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    arms = asyncio.run(run_all(model, tokenizer, args.budget))
    summary = {mode: {
        "median_wall_seconds": round(statistics.median(
            row["wall_seconds"] for row in arms if row["mode"] == mode), 4),
        "runs": sum(row["mode"] == mode for row in arms),
    } for mode in ("pipeline", "barrier")}
    result = {"status": "inference_smoke_only", "model": args.model,
              "frontier_head": "random_initialization", "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "budget": args.budget,
              "state_mode": "auto", "prefix_cache_min_work": 768,
              "load_seconds": round(load_seconds, 3),
              "warmup_seconds": round(warmup_seconds, 3),
              "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
              "summary": summary, "arms": arms}
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


async def run_all(model, tokenizer, budget):
    arms = []
    for mode in ("pipeline", "barrier", "barrier", "pipeline"):
        arms.append(await run_arm(model, tokenizer, mode, budget))
        torch.cuda.synchronize()
    return arms


if __name__ == "__main__":
    main()
