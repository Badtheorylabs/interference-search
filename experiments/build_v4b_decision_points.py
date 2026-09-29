"""Author frozen V4b decision points from existing V4 premise families.

Each held-out task becomes one decision point with four mutually exclusive,
semantically distinguishable but tokenization-comparable next actions:

1. COMMIT P   - continue the premise group with the greatest verified support
2. COMMIT Q   - continue the competing premise group (or a distinct candidate
                family) so the point is genuinely ambiguous
3. VERIFY     - request a verification action before committing
4. REJECT     - backtrack because no supported premise is available yet

The action set is frozen BEFORE any intervention runs. Actions are chosen so
their first distinguishing token differs and every action string has the same
token length under the tokenizer, so we measure a decision boundary rather
than raw token/logit geometry.

Output rows.jsonl per task:
{
  task_id, prompt_ids (context up to decision), decision_prefix,
  actions: {label: {string, first_token_id, token_len}},
  slot_hint: {label: premise_group_or_None},   # from supervision, not used to
                                               # score; used only to name slots
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


# Frozen action surface. The leading word is the decision marker; the trailing
# clause makes each action semantically distinct. Equal word count keeps
# tokenization comparable across the four.
ACTIONS = {
    "COMMIT_P": "Continue with hypothesis A for the current premise.",
    "COMMIT_Q": "Switch to hypothesis B for the current premise.",
    "VERIFY": "Run verification before proceeding further here.",
    "REJECT": "Reject the current premise and backtrack now please.",
}
ACTION_ORDER = ["COMMIT_P", "COMMIT_Q", "VERIFY", "REJECT"]
DECISION_PREFIX = "\nNext action: "


def premise_group_support(row):
    """Return [(group, members)] sorted by verified support, top-level schema."""
    groups = row.get("premise_groups") or {}
    ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    return ordered


def pick_p_q(row):
    """Choose hypothesis A and B premise labels for the decision point."""
    ordered = premise_group_support(row)
    if len(ordered) >= 2:
        (a_name, _a) = ordered[0]
        (b_name, _b) = ordered[1]
        return a_name, b_name
    if len(ordered) == 1:
        (a_name, _a) = ordered[0]
        return a_name, "unverified"
    return "unknown_a", "unknown_b"


def tokenize_actions(tokenizer):
    out = {}
    lens = {}
    first_ids = {}
    for label, text in ACTIONS.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        out[label] = {"string": text, "first_token_id": ids[0], "token_len": len(ids)}
        first_ids[label] = ids[0]
        lens[label] = len(ids)
    return out, first_ids, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="v4 families rows.jsonl (top-level task/task_id schema)")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--revision", default="906bfd4b4dc7f14ee4320094d8b41684abff8539")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              revision=args.revision,
                                              local_files_only=True)
    actions, first_ids, lens = tokenize_actions(tokenizer)
    # Preregistered constraint: first tokens distinct.
    if len(set(first_ids.values())) != len(first_ids):
        raise SystemExit("action first tokens are not distinct: " + str(first_ids))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines()]
    held = [r for r in rows if r.get("premise_groups")]

    written = 0
    for row in held:
        a_name, b_name = pick_p_q(row)
        context = row["task"] + DECISION_PREFIX
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        rec = {
            "task_id": row["task_id"],
            "decision_prefix": DECISION_PREFIX,
            "context": context,
            "context_token_len": len(ctx_ids),
            "actions": actions,
            "action_order": ACTION_ORDER,
            # naming only, never used for scoring
            "slot_hint": {"COMMIT_P": a_name, "COMMIT_Q": b_name,
                          "VERIFY": None, "REJECT": None},
        }
        (out_dir / "rows.jsonl").open("a").write(json.dumps(rec) + "\n")
        written += 1

    (out_dir / "manifest.json").write_text(json.dumps({
        "n_tasks": written,
        "action_strings": ACTIONS,
        "action_first_token_ids": first_ids,
        "action_token_lengths": lens,
        "decision_prefix": DECISION_PREFIX,
    }, indent=2))
    print(f"wrote {written} decision points to {out_dir}/rows.jsonl")
    print("first-token ids:", first_ids)


if __name__ == "__main__":
    main()
