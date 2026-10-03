"""Check whether a stock LLM can propose valid moves for a batched frontier.

This is an inference-only action-format preflight. It does not search to a
solution and does not use the exact solver to rank candidates.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.countdown_actions import CountdownAction, CountdownActionCodec


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
OBJECT = re.compile(r"\{[^{}]*\}")


def parse_action(text, codec, state):
    match = OBJECT.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        action = CountdownAction(int(obj["i"]), int(obj["j"]), str(obj["op"]))
        action_id = codec.index[action]
    except (ValueError, KeyError, TypeError):
        return None
    return action_id if codec.legal(state, action_id) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/native_countdown/validation.jsonl"))
    parser.add_argument("--problems", type=int, default=2)
    parser.add_argument("--states-per-problem", type=int, default=8)
    parser.add_argument("--samples-per-state", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    torch.manual_seed(41)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()[:args.problems]]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
    ).eval()
    codec = CountdownActionCodec(max_numbers=7)
    prompts, metadata = [], []
    for row_index, row in enumerate(rows):
        for state in row["frontier"][:args.states_per_problem]:
            shown = ", ".join(f"{index}={value}" for index, value in enumerate(state))
            message = (
                f"Target: {row['target']}. Current available numbers by zero-based position: {shown}. "
                "Choose exactly one move: two different positions i and j and an operation +, -, *, or /. "
                "The result must be a positive whole number. More moves can follow later. "
                'Reply only as JSON: {"i":0,"op":"+","j":1}.'
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
            )
            for _ in range(args.samples_per_state):
                prompts.append(prompt)
                metadata.append((row_index, tuple(state)))
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.8, top_k=20,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    texts = tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    records = []
    by_state = {}
    for (row_index, state), text in zip(metadata, texts):
        action_id = parse_action(text, codec, state)
        successor = codec.apply(state, action_id) if action_id is not None else None
        by_state.setdefault((row_index, state), []).append(successor)
        records.append({"problem_index": row_index, "state": state, "text": text[:160],
                        "valid": action_id is not None, "successor": successor})
    valid = sum(record["valid"] for record in records)
    distinct = sum(len({item for item in values if item is not None}) for values in by_state.values())
    print(json.dumps({
        "status": "inference-only batched action-format preflight",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "problems": len(rows), "states": len(by_state), "samples": len(records),
        "valid_actions": valid, "valid_rate": valid / max(len(records), 1),
        "distinct_successors_across_states": distinct,
        "mean_distinct_successors_per_state": distinct / max(len(by_state), 1),
        "batch_seconds": round(elapsed, 3),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
