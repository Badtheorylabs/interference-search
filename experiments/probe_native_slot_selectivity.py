"""Probe semantic routing, slot diversity, and failure selectivity."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM


def clipped(tokenizer, text, limit):
    return tokenizer.encode(text, add_special_tokens=False)[:limit]


def build(tokenizer, row, feedback_row, positive_first, marker_on_positive=False,
          include_marker=True):
    negative = row["negatives"][0]
    positive = row["positive"]
    task = clipped(tokenizer, "Coding task:\n" + row["task"] + "\n", 128)
    source_a = positive["source"] if positive_first else negative["source"]
    source_b = negative["source"] if positive_first else positive["source"]
    a = clipped(tokenizer, "Candidate A:\n" + source_a + "\n", 192)
    b = clipped(tokenizer, "Candidate B:\n" + source_b + "\n", 192)
    feedback = "\n".join(
        f"{status}: {detail}" for status, detail in feedback_row["negatives"][0]["tests"])
    evidence = clipped(tokenizer, "Verified execution failure:\n" + feedback + "\n", 128)
    marker = tokenizer.encode("<tool_response>", add_special_tokens=False) if include_marker else []
    instruction = clipped(
        tokenizer, "Which candidate is correct? Answer A or B:\n", 24)
    marker_after_a = (not positive_first) != marker_on_positive
    if marker_after_a:
        ids = task + a + evidence + marker + b + instruction
    else:
        ids = task + a + b + evidence + marker + instruction
    target = tokenizer.encode("A" if positive_first else "B", add_special_tokens=False)[0]
    return (torch.tensor(ids, device="cuda")[None],
            torch.ones(1, len(ids), dtype=torch.long, device="cuda"), target)


def effective_rank(frontier):
    singular = torch.linalg.svdvals(frontier.float())
    probability = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def slot_similarity(frontier):
    normalized = torch.nn.functional.normalize(frontier.float(), dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    slots = cosine.shape[-1]
    off_diagonal = cosine[~torch.eye(slots, dtype=torch.bool, device=cosine.device)]
    return float(off_diagonal.mean())


def load_model(args, tokenizer):
    base = AutoConfig.from_pretrained(args.model, local_files_only=True)
    values = base.to_dict(); values.pop("model_type", None); values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_response>"))
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    model.model.reset_native_output(); model.model.set_native_dtype(torch.float32)
    state = torch.load(args.native_state, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-state", required=True)
    parser.add_argument("--data", default="data/mbpp_same_state_118/rows.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    rows = all_rows[int(len(all_rows) * 0.8):]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args, tokenizer)
    conditions = ("matched", "shuffled", "misassigned", "no_marker", "zero")
    metrics = {name: [] for name in conditions}
    gate_write, gate_read, ref_match = [], [], []
    for index, row in enumerate(rows):
        positive_first = index % 2 == 0
        shuffled = rows[(index + 7) % len(rows)]
        specs = {
            "matched": (row, False, True, "inhibit"),
            "shuffled": (shuffled, False, True, "inhibit"),
            "misassigned": (row, True, True, "inhibit"),
            "no_marker": (row, False, False, "inhibit"),
            "zero": (row, False, True, "zero"),
        }
        for name, (feedback_row, marker_on_positive, include_marker, coupling) in specs.items():
            ids, mask, target = build(
                tokenizer, row, feedback_row, positive_first,
                marker_on_positive, include_marker)
            output = model(input_ids=ids, attention_mask=mask,
                           use_cache=False, coupling=coupling)
            logp = output.logits[:, -1].float().log_softmax(dim=-1)[0, target]
            record = {"task_id": row["task_id"], "target_logp": float(logp),
                      "effective_rank": effective_rank(output.frontier_state[0]),
                      "mean_slot_cosine": slot_similarity(output.frontier_state[0])}
            metrics[name].append(record)
            if name == "matched":
                gate_write.append(output.slot_write_gate[0].mean(dim=-1).cpu().tolist())
                gate_read.append(output.token_read_gate[0, :, 0].cpu().tolist())
                ref_match.append(output.refutation_match[0].mean(dim=-1).cpu().tolist())
    summary = {}
    for name, values in metrics.items():
        summary[name] = {
            "mean_target_logp": statistics.mean(row["target_logp"] for row in values),
            "mean_effective_rank": statistics.mean(row["effective_rank"] for row in values),
            "mean_slot_cosine": statistics.mean(row["mean_slot_cosine"] for row in values),
        }
    matched = [row["target_logp"] for row in metrics["matched"]]
    for name in ("shuffled", "misassigned", "no_marker", "zero"):
        other = [row["target_logp"] for row in metrics[name]]
        summary[name]["matched_minus_condition"] = statistics.mean(
            left - right for left, right in zip(matched, other))
    layers = len(gate_write[0])
    layer_metrics = {
        "mean_write_gate": [statistics.mean(row[layer] for row in gate_write)
                            for layer in range(layers)],
        "mean_read_gate": [statistics.mean(row[layer] for row in gate_read)
                           for layer in range(layers)],
        "mean_refutation_match": [statistics.mean(row[layer] for row in ref_match)
                                  for layer in range(layers)],
    }
    report = {"status": "native slot selectivity probe", "tasks": len(rows),
              "summary": summary, "layer_metrics": layer_metrics, "rows": metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tasks": len(rows),
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
