"""Project paired SWE-Smith rollouts into real Interference Search frontiers."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


REVISION = "08e109b4a59eaeebf80e4675cd125d42e7ac99a4"
FRACTIONS = (0.25, 0.5, 0.75)


def text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item.get("content", "")))
                          for item in content if isinstance(item, dict))
    return str(content or "")


def normalize(value):
    return re.sub(r"\s+", " ", value).strip()


def digest(value):
    return hashlib.sha256(normalize(value).encode()).hexdigest()


def assistant_action(message):
    calls = message.get("tool_calls") or []
    if calls:
        return json.dumps(calls, sort_keys=True, separators=(",", ":"))
    content = text(message.get("content"))
    if "<function=" in content or "```bash" in content or "<tool_call>" in content:
        return content
    return None


def executable_steps(messages):
    steps = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        action = assistant_action(message)
        if not action:
            continue
        observation = None
        for following in messages[index + 1:index + 4]:
            if following.get("role") in {"tool", "user"}:
                candidate = text(following.get("content"))
                if candidate.strip():
                    observation = candidate
                    break
        if observation:
            prefix = messages[max(0, index - 8):index]
            state = "\n".join(f"{item.get('role')}: {text(item.get('content'))}"
                              for item in prefix)
            steps.append({"state": state[-12000:], "action": action[-6000:],
                          "observation": observation[-6000:]})
    return steps


def task_text(messages):
    for message in messages:
        if message.get("role") == "user":
            value = text(message.get("content"))
            if value.strip():
                return value[-16000:]
    return ""


def repo_family(instance_id):
    return instance_id.split(".", 1)[0]


def split_for(family):
    value = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def select_balanced(records, limit=8):
    good = sorted((record for record in records if record["resolved"]),
                  key=lambda item: item["trajectory_id"])
    bad = sorted((record for record in records if not record["resolved"]),
                 key=lambda item: item["trajectory_id"])
    selected = []
    while len(selected) < limit and (good or bad):
        if good:
            selected.append(good.pop(0))
        if len(selected) < limit and bad:
            selected.append(bad.pop(0))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/real_swe_frontier_v0"))
    args = parser.parse_args()
    groups = defaultdict(list)
    source_files = sorted(args.source.glob("*.parquet"))
    for source_file in source_files:
        table = pq.read_table(source_file, columns=[
            "messages", "instance_id", "resolved", "model", "traj_id", "patch"])
        for raw in table.to_pylist():
            messages = json.loads(raw["messages"])
            steps = executable_steps(messages)
            if not steps:
                continue
            groups[raw["instance_id"]].append({
                "trajectory_id": raw["traj_id"], "model": raw["model"],
                "resolved": bool(raw["resolved"]), "patch": raw["patch"],
                "task": task_text(messages), "steps": steps,
            })
    outputs = {name: [] for name in ("train", "validation", "test")}
    dropped = defaultdict(int)
    for instance_id, trajectories in sorted(groups.items()):
        if not any(item["resolved"] for item in trajectories) or not any(
                not item["resolved"] for item in trajectories):
            dropped["not_contrastive"] += 1
            continue
        trajectories = select_balanced(trajectories)
        family = repo_family(instance_id)
        split = split_for(family)
        for fraction in FRACTIONS:
            branches = []
            seen = set()
            for trajectory in trajectories:
                index = min(len(trajectory["steps"]) - 1,
                            int(fraction * len(trajectory["steps"])))
                step = trajectory["steps"][index]
                transition_hash = digest(step["action"] + "\n" + step["observation"])
                if transition_hash in seen:
                    continue
                seen.add(transition_hash)
                branches.append({
                    "trajectory_id": trajectory["trajectory_id"],
                    "model": trajectory["model"], "state": step["state"],
                    "action": step["action"], "observation": step["observation"],
                    "resolved": trajectory["resolved"],
                    "state_hash": digest(step["state"]),
                    "transition_hash": transition_hash,
                    "patch_hash": digest(trajectory["patch"]) if trajectory["patch"] else None,
                })
            if len(branches) < 2 or not any(branch["resolved"] for branch in branches) or not any(
                    not branch["resolved"] for branch in branches):
                dropped["projection_not_contrastive"] += 1
                continue
            row_id = f"real-swe-frontier-{split}-{len(outputs[split]):06d}"
            outputs[split].append({
                "id": row_id, "split": split,
                "source": {"dataset": "SWE-bench/SWE-smith-trajectories",
                           "revision": REVISION, "license": "MIT"},
                "instance_id": instance_id, "repo_family": family,
                "task": trajectories[0]["task"], "fraction": fraction,
                "branches": branches,
            })
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        path = args.out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                for row in rows))
    golden = (outputs["train"][:1] + outputs["validation"][:1] + outputs["test"][:1])
    (args.out / "golden_examples.jsonl").write_text(
        "".join(json.dumps(row, indent=None) + "\n" for row in golden))
    report = {
        "source_files": len(source_files), "source_rows": sum(len(v) for v in groups.values()),
        "contrastive_instances": len({row["instance_id"] for rows in outputs.values() for row in rows}),
        "rows": {split: len(rows) for split, rows in outputs.items()},
        "branches": {split: sum(len(row["branches"]) for row in rows)
                     for split, rows in outputs.items()},
        "resolved_branches": {split: sum(branch["resolved"] for row in rows for branch in row["branches"])
                              for split, rows in outputs.items()},
        "dropped": dict(dropped),
        "repo_families": {split: len({row["repo_family"] for row in rows})
                          for split, rows in outputs.items()},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
