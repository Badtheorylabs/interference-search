"""Inspect one frozen MBPP task's raw generations and sandbox outcomes."""

import argparse
import asyncio
import json
from pathlib import Path

from interference_search.mbpp_async_domain import MBPPAsyncDomain, extract_code
from interference_search.sandboxed_python import BubblewrapPython
from interference_search.vllm_code_backend import VLLMCodeBackend


ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    task = next(row for row in json.loads((ROOT / "data/mbpp_sanitized.json").read_text())
                if row["task_id"] == args.task_id)
    backend = VLLMCodeBackend(args.model, max_tokens=args.max_new_tokens,
                              enforce_eager=args.enforce_eager)
    sandbox = BubblewrapPython()
    domain = MBPPAsyncDomain(backend, sandbox)
    try:
        root = domain.start(task)
        prompts = [domain._prompt(root, task, variant) for variant in range(2)]
        generations = await backend.generate_batch(prompts, [11, 13])
        results = []
        for generated in generations:
            source = extract_code(generated.text)
            verdict = await sandbox.run(source, task["test_list"], task["test_imports"])
            results.append({"raw": generated.text, "source": source,
                            "output_tokens": generated.output_tokens,
                            "tests": verdict.tests,
                            "sandbox_error": verdict.sandbox_error})
        print(json.dumps({"task_id": args.task_id, "prompt": task["prompt"],
                          "results": results}, indent=2))
    finally:
        await backend.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-id", type=int, default=726)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
