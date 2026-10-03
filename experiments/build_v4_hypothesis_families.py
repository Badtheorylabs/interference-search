"""Build V4 hypothesis-family datasets with verifier-certified metadata.

Every candidate carries:
  - a behavioral outcome vector (pass/fail per test)
  - an error class for failures
  - an equivalence class (candidates with identical outcome vectors)
  - a premise group (same error class over overlapping failing tests)

All labels come from real sandbox execution. Nothing is guessed by a model.
"""

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
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


def outcome_vector(tests):
    return [status == "pass" for status, _ in tests]


def error_class(tests):
    for status, detail in tests:
        if status != "pass" and detail:
            head = detail.split(":", 1)[0].strip()
            if head:
                return head
    return "UnknownError"


def failing_mask(vector):
    return ",".join(str(i) for i, ok in enumerate(vector) if not ok)


def mutations(source):
    replacements = [
        (" + ", " - "), (" - ", " + "), (" * ", " + "),
        (" == ", " != "), (" <= ", " < "), (" >= ", " > "),
        ("True", "False"), ("False", "True"),
    ]
    values = []
    for old, new in replacements:
        if old in source:
            values.append(source.replace(old, new, 1))
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("return "):
            changed = list(lines)
            indent = line[:len(line) - len(line.lstrip())]
            changed[index] = indent + "return None"
            values.append("\n".join(changed))
            break
    return list(dict.fromkeys(value for value in values if value != source))


async def build_row(backend, sandbox, task, samples, seed):
    prompt = backend.format_prompt(fresh_instruction(task))
    generations = await backend.generate_batch(
        [prompt] * samples, [seed + offset for offset in range(samples)])
    raw_sources = [extract_code(generation.text) for generation in generations]
    reference = task["code"]

    # Collect distinct candidates: samples plus reference. Mutations run only
    # when sampled outcomes are homogeneous, keeping sandbox cost bounded.
    ordered = []
    seen = set()

    def offer(source):
        if not source:
            return
        digest = hashlib.sha1(source.encode()).hexdigest()
        if digest in seen:
            return
        seen.add(digest)
        ordered.append(source)

    for source in raw_sources:
        offer(source)
    offer(reference)

    outcomes = await asyncio.gather(*(
        sandbox.run(source, task["test_list"], task.get("test_imports", []))
        for source in ordered))
    all_passed = [outcome.all_passed for outcome in outcomes]

    if all(all_passed) or not any(all_passed):
        mutated = []
        mutation_sources = [reference] + [s for s in raw_sources if s][:2]
        for source in mutation_sources:
            mutated.extend(mutations(source))
        mutated = list(dict.fromkeys(
            source for source in mutated if hashlib.sha1(
                source.encode()).hexdigest() not in seen))
        mutated = mutated[:16]
        if mutated:
            mutated_outcomes = await asyncio.gather(*(
                sandbox.run(source, task["test_list"],
                            task.get("test_imports", []))
                for source in mutated))
            for source, outcome in zip(mutated, mutated_outcomes):
                ordered.append(source)
                outcomes.append(outcome)

    candidates = []
    for source, outcome in zip(ordered, outcomes):
        vector = outcome_vector(outcome.tests)
        candidates.append({
            "source": source,
            "outcome_vector": vector,
            "all_passed": outcome.all_passed,
            "error_class": None if outcome.all_passed else error_class(outcome.tests),
            "failing_mask": None if outcome.all_passed else failing_mask(vector),
            "value": sum(vector) / max(len(vector), 1),
        })

    positives = [item for item in candidates if item["all_passed"]]
    negatives = [item for item in candidates if not item["all_passed"]]
    if not positives or not negatives:
        return None

    # Verifier-certified equivalence: identical behavioral outcome vectors.
    class_by_vector = defaultdict(list)
    for index, item in enumerate(candidates):
        class_by_vector[tuple(item["outcome_vector"])].append(index)
    equivalence_classes = [group for group in class_by_vector.values()
                           if len(group) >= 2]

    # Premise families: failures sharing an error class and overlapping
    # failing tests belong to the same premise region. Passing candidates
    # form a single verified family.
    premise_groups = defaultdict(list)
    for index, item in enumerate(candidates):
        if item["all_passed"]:
            premise_groups["verified"].append(index)
        else:
            premise_groups[
                f"{item['error_class']}|{item['failing_mask']}"].append(index)
    premise_groups = {key: group for key, group in premise_groups.items()
                      if len(group) >= 1}

    distinct_masks = len({tuple(item["outcome_vector"]) for item in candidates})
    if distinct_masks < 2:
        return None

    return {
        "task_id": task["task_id"], "task": task["prompt"],
        "tests": task["test_list"],
        "candidates": candidates,
        "equivalence_classes": equivalence_classes,
        "premise_groups": premise_groups,
        "num_passing": len(positives), "num_failing": len(negatives),
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
        "status": "V4 hypothesis families",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "start": args.start, "tasks": len(tasks), "rows": len(rows),
        "samples_per_state": args.samples,
        "rows_with_four_candidates": sum(
            1 for row in rows if len(row["candidates"]) >= 4),
        "rows_with_equivalence": sum(
            1 for row in rows if row["equivalence_classes"]),
        "rows_with_family_ge2": sum(
            1 for row in rows
            if any(len(group) >= 2 for group in
                   row["premise_groups"].values())),
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
    parser.add_argument("--start", type=int, default=118)
    parser.add_argument("--count", type=int, default=309)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
