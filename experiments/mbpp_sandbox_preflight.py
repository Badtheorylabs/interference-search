"""Freeze a model-blind MBPP slice and verify reference code in the sandbox."""

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

from interference_search.sandboxed_python import BubblewrapPython


ROOT = Path(__file__).resolve().parents[1]


async def main_async(args):
    source = ROOT / "data/mbpp_sanitized.json"
    raw = source.read_bytes()
    rows = json.loads(raw)
    sample = random.Random(args.seed).sample(rows, args.screen)
    runner = BubblewrapPython()
    semaphore = asyncio.Semaphore(args.workers)

    async def check(row):
        async with semaphore:
            result = await runner.run(row["code"], row["test_list"],
                                      row.get("test_imports", []))
        return {"task_id": row["task_id"], "passed": result.passed,
                "total": len(row["test_list"]), "all_passed": result.all_passed,
                "sandbox_error": result.sandbox_error,
                "wall_ms": round(result.wall_seconds * 1000, 3)}

    checks = await asyncio.gather(*(check(row) for row in sample))
    ids = [item["task_id"] for item in checks if item["all_passed"]][:args.tasks]
    if len(ids) != args.tasks:
        raise RuntimeError("too few reference programs pass sandboxed tests")
    result = {"dataset": "mbpp_sanitized.json",
              "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "selection": "seeded random sample, then reference-pass filter; no model outputs",
              "seed": args.seed, "screen": args.screen, "task_ids": ids,
              "reference_preflight": checks}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"task_ids": ids, "reference_passed": sum(
        item["all_passed"] for item in checks), "screen": args.screen,
                      "dataset_sha256": result["dataset_sha256"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--screen", type=int, default=32)
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
