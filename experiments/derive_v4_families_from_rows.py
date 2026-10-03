"""Derive V4 hypothesis-family metadata for existing same-state rows.

Recomputes outcome vectors, error classes, verifier equivalence classes, and
premise groups from the stored per-test outcomes. Adds the benchmark reference
solution when it differs from stored candidates and passes the task tests.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


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


def candidate(source, tests):
    vector = outcome_vector(tests)
    all_passed = all(vector)
    return {
        "source": source,
        "outcome_vector": vector,
        "all_passed": all_passed,
        "error_class": None if all_passed else error_class(tests),
        "failing_mask": None if all_passed else failing_mask(vector),
        "value": sum(vector) / max(len(vector), 1),
    }


def annotate(row):
    candidates = [candidate(row["positive"]["source"], row["positive"]["tests"])]
    for item in row["negatives"]:
        candidates.append(candidate(item["source"], item["tests"]))

    class_by_vector = defaultdict(list)
    for index, item in enumerate(candidates):
        class_by_vector[tuple(item["outcome_vector"])].append(index)
    equivalence_classes = [group for group in class_by_vector.values()
                           if len(group) >= 2]

    premise_groups = defaultdict(list)
    for index, item in enumerate(candidates):
        if item["all_passed"]:
            premise_groups["verified"].append(index)
        else:
            premise_groups[
                f"{item['error_class']}|{item['failing_mask']}"].append(index)

    return {
        "task_id": row["task_id"], "task": row["task"], "tests": row["tests"],
        "candidates": candidates,
        "equivalence_classes": equivalence_classes,
        "premise_groups": dict(premise_groups),
        "num_passing": sum(1 for item in candidates if item["all_passed"]),
        "num_failing": sum(1 for item in candidates if not item["all_passed"]),
        "samples": row.get("samples"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.src.read_text().splitlines()]
    annotated = [annotate(row) for row in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "rows.jsonl"
    with data_path.open("w") as handle:
        for row in annotated:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "status": "derived V4 families from stored rows",
        "rows": len(annotated),
        "rows_with_four_candidates": sum(
            1 for row in annotated if len(row["candidates"]) >= 4),
        "rows_with_equivalence": sum(
            1 for row in annotated if row["equivalence_classes"]),
        "rows_with_family_ge2": sum(
            1 for row in annotated
            if any(len(group) >= 2 for group in row["premise_groups"].values())),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
