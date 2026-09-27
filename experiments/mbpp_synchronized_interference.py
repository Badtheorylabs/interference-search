"""Paired agentic-code gate for synchronized Interference Search."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from pathlib import Path

from interference_search.mbpp_async_domain import MBPPAsyncDomain, canonical_source
from interference_search.native_interference_search import synchronized_search
from interference_search.sandboxed_python import BubblewrapPython
from interference_search.vllm_code_backend import VLLMCodeBackend


ROOT = Path(__file__).resolve().parents[1]


def load_tasks(manifest_path, limit):
    manifest = json.loads(Path(manifest_path).read_text())
    raw = (ROOT / "data/mbpp_sanitized.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("MBPP dataset differs from frozen manifest")
    by_id = {row["task_id"]: row for row in json.loads(raw)}
    ids = manifest["task_ids"][:limit]
    return [by_id[task_id] for task_id in ids], manifest


def backend_receipt(backend):
    return (backend.started_requests, backend.requests,
            backend.started_prompt_tokens, backend.prompt_tokens,
            backend.output_tokens, backend.request_wall_seconds)


async def trial(backend, task, arm, args):
    if arm == "interference":
        config = {"fresh_candidates": 4, "revision_candidates": 1,
                  "max_depth": 3, "width": 2, "expansion_budget": 5,
                  "merge": True}
    else:
        config = {"fresh_candidates": 1, "revision_candidates": 1,
                  "max_depth": 8, "width": 1, "expansion_budget": 8,
                  "merge": False}
    domain = MBPPAsyncDomain(
        backend, BubblewrapPython(),
        fresh_candidates=config["fresh_candidates"],
        revision_candidates=config["revision_candidates"],
        max_depth=config["max_depth"])
    before = backend_receipt(backend)
    result = await asyncio.wait_for(synchronized_search(
        domain, task, expansion_budget=config["expansion_budget"],
        width=config["width"], execution_workers=8,
        merge=config["merge"]), timeout=args.task_timeout)
    after = backend_receipt(backend)
    candidates = [transition["child"] for event in result.events
                  for transition in event["transitions"]]
    sandbox_errors = [state.outcome.sandbox_error for state in candidates
                      if state.outcome and state.outcome.sandbox_error]
    if sandbox_errors:
        raise RuntimeError(f"sandbox failed: {sandbox_errors[0]}")
    best = max((state.outcome.passed for state in candidates if state.outcome), default=0)
    solved_source = result.solution.source if result.solved else None
    return {
        "task_id": task["task_id"], "arm": arm, "solved": result.solved,
        "wall_seconds": result.wall_seconds, "rounds": result.rounds,
        "expanded": result.expanded, "proposed": result.proposed,
        "executions": result.executions, "merged_away": result.merged_away,
        "model_requests_started": after[0] - before[0],
        "model_requests_completed": after[1] - before[1],
        "started_prompt_tokens": after[2] - before[2],
        "prompt_tokens": after[3] - before[3],
        "output_tokens": after[4] - before[4],
        "sum_model_wall_seconds": after[5] - before[5],
        "best_visible_tests": best, "visible_tests": len(task["test_list"]),
        "unique_programs": len({canonical_source(state.source) for state in candidates}),
        "solution_sha256": (hashlib.sha256(solved_source.encode()).hexdigest()
                            if solved_source else None),
        "config": config,
    }


async def measure(args):
    tasks, manifest = load_tasks(args.manifest, args.limit)
    backend = VLLMCodeBackend(
        args.model, max_tokens=args.max_new_tokens,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization)
    rows = []
    try:
        for index, task in enumerate(tasks):
            order = ("interference", "normal") if index % 2 == 0 else (
                "normal", "interference")
            for arm in order:
                row = await trial(backend, task, arm, args)
                row["run_order"] = len(rows)
                rows.append(row)
                print(json.dumps({key: row[key] for key in (
                    "task_id", "arm", "solved", "wall_seconds",
                    "model_requests_started", "output_tokens", "best_visible_tests")}),
                    flush=True)
                Path(args.output).write_text(json.dumps({
                    "status": "partial", "rows": rows}, indent=2) + "\n")
    finally:
        await backend.shutdown()
    groups = {}
    for arm in ("interference", "normal"):
        subset = [row for row in rows if row["arm"] == arm]
        groups[arm] = {
            "solved": sum(row["solved"] for row in subset), "total": len(subset),
            **{f"mean_{key}": statistics.mean(row[key] for row in subset)
               for key in ("wall_seconds", "model_requests_started", "output_tokens",
                           "executions", "merged_away", "best_visible_tests")},
        }
    return {
        "status": "complete", "model": args.model,
        "dataset_sha256": manifest["dataset_sha256"],
        "task_ids": [task["task_id"] for task in tasks],
        "max_new_tokens": args.max_new_tokens,
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "groups": groups, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(
        ROOT / "results/mbpp_runtime_manifest_replication64.json"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--task-timeout", type=float, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = asyncio.run(measure(args))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
