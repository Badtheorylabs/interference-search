"""Paired MBPP pipeline/barrier benchmark on one pinned vLLM backend.

Tasks are model-blind frozen by mbpp_sandbox_preflight.py. Both arms use the
same prompts, greedy generation, bubblewrap executor, model batch capacity,
tool concurrency, and expansion budget. They run in alternating order to
reduce shared engine warm-cache bias.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from pathlib import Path

from interference_search.async_execution import AsyncExecutionSearch
from interference_search.mbpp_async_domain import MBPPAsyncDomain, canonical_source
from interference_search.sandboxed_python import BubblewrapPython
from interference_search.vllm_code_backend import VLLMCodeBackend


ROOT = Path(__file__).resolve().parents[1]


def load_frozen_tasks(manifest_path: Path, limit: int | None,
                      selected_ids: list[int] | None):
    manifest = json.loads(manifest_path.read_text())
    raw = (ROOT / "data/mbpp_sanitized.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("MBPP dataset differs from the frozen manifest")
    by_id = {row["task_id"]: row for row in json.loads(raw)}
    ids = manifest["task_ids"][:limit]
    if selected_ids is not None:
        if not set(selected_ids).issubset(ids):
            raise ValueError("selected tasks must belong to the frozen manifest")
        ids = [task_id for task_id in ids if task_id in selected_ids]
    return [by_id[task_id] for task_id in ids], manifest


async def run_trial(backend, task, mode, args):
    domain = MBPPAsyncDomain(backend, BubblewrapPython(),
                             fresh_candidates=args.fresh_candidates,
                             revision_candidates=args.revision_candidates,
                             max_depth=args.max_depth)
    scheduler = AsyncExecutionSearch(
        model_batch_size=args.model_batch_size,
        execution_workers=args.execution_workers,
        max_inflight_model_batches=(args.max_inflight_model_batches
                                    if mode == "pipeline" else 1),
        mode=mode, merge_states=True, dedupe_executions=True,
        batch_wait_seconds=args.batch_wait_ms / 1000,
    )
    before = (backend.started_requests, backend.requests,
              backend.started_prompt_tokens, backend.prompt_tokens,
              backend.output_tokens, backend.request_wall_seconds,
              backend.parity_mismatches, backend.cancelled_requests)
    result = await asyncio.wait_for(scheduler.run(domain, task, args.expansion_budget),
                                    timeout=args.task_timeout)
    after = (backend.started_requests, backend.requests,
             backend.started_prompt_tokens, backend.prompt_tokens,
             backend.output_tokens, backend.request_wall_seconds,
             backend.parity_mismatches, backend.cancelled_requests)
    candidate_states = [node.state for node in result.graph.values()
                        if node.state.source is not None]
    sandbox_errors = [state.outcome.sandbox_error for state in candidate_states
                      if state.outcome and state.outcome.sandbox_error]
    if sandbox_errors:
        raise RuntimeError(f"sandbox failed during MBPP scoring: {sandbox_errors[0]}")
    source_hash = None
    if result.solved:
        source_hash = hashlib.sha256(result.solution.source.encode()).hexdigest()
    return {"task_id": task["task_id"], "mode": mode, "solved": result.solved,
            "wall_seconds": round(result.wall_seconds, 4),
            "expansions": result.expansions,
            "model_batches": result.proposal_batches,
            "proposed_actions": result.proposed_actions,
            "model_requests": after[0] - before[0],
            "completed_model_requests": after[1] - before[1],
            "started_prompt_tokens": after[2] - before[2],
            "prompt_tokens": after[3] - before[3],
            "output_tokens": after[4] - before[4],
            "sum_request_wall_seconds": round(after[5] - before[5], 4),
            "same_prompt_output_mismatches": after[6] - before[6],
            "cancelled_model_requests": after[7] - before[7],
            "execution_requests": result.execution_requests,
            "execution_calls": result.execution_calls,
            "execution_coalesced": result.execution_coalesced,
            "execution_cache_hits": result.execution_cache_hits,
            "cancelled_executions": result.cancelled_executions,
            "state_merges": result.state_merges,
            "unique_programs": len({canonical_source(state.source)
                                    for state in candidate_states}),
            "syntax_errors": sum(
                any(status == "load_error" and detail.startswith("SyntaxError")
                    for status, detail in state.outcome.tests)
                for state in candidate_states if state.outcome),
            "sandbox_errors": 0,
            "best_visible_tests": max((state.outcome.passed for state in candidate_states
                                       if state.outcome), default=0),
            "visible_tests": len(task["test_list"]),
            "solution_source_sha256": source_hash}


async def measure(args):
    tasks, manifest = load_frozen_tasks(Path(args.manifest), args.limit,
                                       args.task_ids)
    backend = VLLMCodeBackend(args.model, max_tokens=args.max_new_tokens,
                              enforce_eager=args.enforce_eager,
                              gpu_memory_utilization=args.gpu_memory_utilization)
    rows = []
    try:
        for index, task in enumerate(tasks):
            order = ("pipeline", "barrier") if index % 2 == 0 else (
                "barrier", "pipeline")
            for mode in order:
                row = await run_trial(backend, task, mode, args)
                row["run_order"] = len(rows)
                rows.append(row)
                print(json.dumps({key: row[key] for key in (
                    "task_id", "mode", "solved", "wall_seconds", "model_requests",
                    "output_tokens", "execution_calls")}), flush=True)
                if args.output:
                    Path(args.output).write_text(json.dumps({
                        "status": "partial", "config": vars(args), "rows": rows,
                    }, indent=2) + "\n")
    finally:
        await backend.shutdown()
    groups = []
    for mode in ("pipeline", "barrier"):
        subset = [row for row in rows if row["mode"] == mode]
        groups.append({"mode": mode, "solved": sum(row["solved"] for row in subset),
                       "total": len(subset),
                       **{f"mean_{key}": round(statistics.mean(row[key] for row in subset), 4)
                          for key in ("wall_seconds", "model_requests",
                                      "completed_model_requests", "started_prompt_tokens",
                                      "prompt_tokens", "output_tokens",
                                      "execution_calls", "expansions")}})
    return {"status": "complete", "dataset_sha256": manifest["dataset_sha256"],
            "task_ids": [task["task_id"] for task in tasks],
            "config": {**vars(args), "VLLM_BATCH_INVARIANT":
                       os.environ.get("VLLM_BATCH_INVARIANT", "0")},
            "groups": groups, "rows": rows,
            "same_prompt_output_mismatches_total": backend.parity_mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "results/mbpp_runtime_manifest_16.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fresh-candidates", type=int, default=2)
    parser.add_argument("--revision-candidates", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--expansion-budget", type=int, default=6)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--max-inflight-model-batches", type=int, default=2)
    parser.add_argument("--execution-workers", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=0.0)
    parser.add_argument("--task-timeout", type=float, default=180.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = asyncio.run(measure(args))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "groups": result["groups"],
                      "same_prompt_output_mismatches_total":
                      result["same_prompt_output_mismatches_total"]}, indent=2))


if __name__ == "__main__":
    main()
