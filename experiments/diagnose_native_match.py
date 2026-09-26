"""Inspect refutation affinity on a saved native Countdown pilot checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.native_lm import NativeFrontierLM
from train_native_countdown_pilot import MODEL, REVISION, CODEC, pack_row, read_rows, to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/native_countdown/validation.jsonl"))
    args = parser.parse_args()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
    )
    model = NativeFrontierLM(base, actions=len(CODEC.actions)).to(device="cuda", dtype=torch.bfloat16).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    row = read_rows(args.data)[0]
    data = to_device(pack_row(row, tokenizer), "cuda")
    with torch.no_grad():
        output = model(
            task_ids=data["task_ids"], task_mask=data["task_mask"],
            state_ids=data["state_ids"], state_token_mask=data["state_token_mask"],
            state_mask=data["state_mask"], refutation_ids=data["refutation_ids"],
            refutation_token_mask=data["refutation_token_mask"],
            refutation_mask=data["refutation_mask"], action_mask=data["action_mask"],
        ).frontier
        refs = model.state_encoder(data["refutation_ids"], data["refutation_token_mask"])
        query = F.normalize(model.frontier_cell.action_query(output.predicted_successors).float(), dim=-1)
        keys = F.normalize(model.frontier_cell.refutation_key(refs).float(), dim=-1)
        affinities = torch.einsum("bwah,brh->bwar", query, keys)
        logits = 4.0 * affinities - model.frontier_cell.match_logit_bias.float()
        valid = data["action_mask"]
        positive = data["refutation_targets"].bool() & valid
        negative = valid & ~positive
        report = {
            "status": "diagnostic only",
            "checkpoint": str(args.checkpoint),
            "match_logit_bias": float(model.frontier_cell.match_logit_bias),
            "positive_affinity_mean": float(affinities.squeeze(-1)[positive].float().mean()),
            "negative_affinity_mean": float(affinities.squeeze(-1)[negative].float().mean()),
            "affinity_min": float(affinities.float().min()),
            "affinity_max": float(affinities.float().max()),
            "positive_match_mean": float(output.action_refutation_match[positive].float().mean()),
            "negative_match_mean": float(output.action_refutation_match[negative].float().mean()),
            "query_norm_mean": float(query.float().norm(dim=-1).mean()),
            "key_norm_mean": float(keys.float().norm(dim=-1).mean()),
            "positive_logit_mean": float(logits.squeeze(-1)[positive].float().mean()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
