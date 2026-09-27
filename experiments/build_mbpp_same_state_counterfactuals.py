"""Build verified same-state coding counterfactuals for native midtraining."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from interference_search.mbpp_async_domain import extract_code
from interference_search.sandboxed_python import BubblewrapPython
from interference_search.vllm_code_backend import VLLMCodeBackend


ROOT = Path(__file__).resolve().parents[1]


def fresh_instruction(task):
    return (
        f"Write the Python function described here:\n{task['prompt']}\n"
        f"It must pass this example:\n{task['test_list'][0]}\n"
        "Use a general implementation that handles edge cases. "
        "Return only Python source code, with no explanation.")


def revision_instruction(task, source, outcome):
    feedback = "\n".join(
        f"{status.upper()} {test}: {detail}" for test, (status, detail)
        in zip(task["test_list"], outcome.tests))
    return (
        f"Fix this Python function for the task:\n{task['prompt']}\n"
        f"Current code:\n```python\n{source[:5000]}\n```\n"
        f"Verified test feedback:\n{feedback[:5000]}\n"
        "Return only corrected Python source code, with no explanation.")


async def execute_candidates(sandbox, task, generations):
    sources = [extract_code(generation.text) for generation in generations]
    outcomes = await asyncio.gather(*(
        sandbox.run(source, task["test_list"], task.get("test_imports", []))
        for source in sources))
    return sources, outcomes


def candidate(source, outcome):
    return {"source": source, "tests": [list(test) for test in outcome.tests],
            "passed": outcome.passed, "all_passed": outcome.all_passed}


async def build_row(backend, sandbox, task, samples, seed):
    prompt = backend.format_prompt(fresh_instruction(task))
    generations = await backend.generate_batch(
        [prompt] * samples, [seed + offset for offset in range(samples)])
    sources, outcomes = await execute_candidates(sandbox, task, generations)
    positives = [index for index, outcome in enumerate(outcomes) if outcome.all_passed]
    negatives = [index for index, outcome in enumerate(outcomes) if not outcome.all_passed]
    state = {"kind": "root", "prompt": prompt}

    if not positives and negatives:
        best = max(negatives, key=lambda index: outcomes[index].passed)
        prompt = backend.format_prompt(revision_instruction(
            task, sources[best], outcomes[best]))
        generations = await backend.generate_batch(
            [prompt] * samples,
            [seed + 1000 + offset for offset in range(samples)])
        revision_sources, revision_outcomes = await execute_candidates(
            sandbox, task, generations)
        positives = [index for index, outcome in enumerate(revision_outcomes)
                     if outcome.all_passed]
        negatives = [index for index, outcome in enumerate(revision_outcomes)
                     if not outcome.all_passed]
        state = {"kind": "revision", "prompt": prompt,
                 "parent_source": sources[best],
                 "parent_tests": [list(test) for test in outcomes[best].tests]}
        sources, outcomes = revision_sources, revision_outcomes

    if not positives or not negatives:
        return None
    return {
        "task_id": task["task_id"], "task": task["prompt"],
        "tests": task["test_list"], "state": state,
        "positive": candidate(sources[positives[0]], outcomes[positives[0]]),
        "negatives": [candidate(sources[index], outcomes[index])
                      for index in negatives],
        "samples": samples,
    }


async def main_async(args):
    raw = (ROOT / "data/mbpp_sanitized.json").read_bytes()
    tasks = json.loads(raw)[args.start:args.start + args.count]
    backend = VLLMCodeBackend(
        args.model, max_tokens=args.max_new_tokens,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature, top_p=args.top_p)
    sandbox = BubblewrapPython()
    rows = []
    try:
        for index, task in enumerate(tasks):
            row = await build_row(
                backend, sandbox, task, args.samples,
                args.seed + task["task_id"] * 100)
            if row is not None:
                rows.append(row)
            print(json.dumps({"tasks": index + 1, "rows": len(rows),
                              "task_id": task["task_id"]}), flush=True)
    finally:
        await backend.shutdown()
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "rows.jsonl"
    with data_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "status": "verified same-state MBPP counterfactuals",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "start": args.start, "tasks": len(tasks), "rows": len(rows),
        "samples_per_state": args.samples,
        "root_rows": sum(row["state"]["kind"] == "root" for row in rows),
        "revision_rows": sum(row["state"]["kind"] == "revision" for row in rows),
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "model_requests": backend.requests,
        "output_tokens": backend.output_tokens,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
