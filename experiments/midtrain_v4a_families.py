"""V4a-families: hypothesis-set supervision with family and equivalence targets.

Changes from the V4a admission run:
  - candidate selection is stratified by premise group and behavioral mask;
  - supervised within-family / between-family embedding margins;
  - supervised equivalence collapse (matched slots of equivalent candidates
    converge and agree), so certified duplicates reduce slot multiplicity;
  - evaluation reports family margin and retrieval multiplicity reduction.
"""

import argparse
import itertools
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM


def model_config(path, tokenizer):
    base = AutoConfig.from_pretrained(path, local_files_only=True)
    values = base.to_dict(); values.pop("model_type", None); values.pop("architectures", None)
    return InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_call>"))


def select_candidates(row, rng, count=4):
    """Stratified across behavioral masks so every row carries genuinely
    distinct hypotheses: at least one passing candidate and two masks."""
    candidates = list(row["candidates"])
    groups = {}
    for item in candidates:
        groups.setdefault(tuple(item["outcome_vector"]), []).append(item)
    keys = list(groups)
    rng.shuffle(keys)
    passing = [key for key in keys if all(groups[key][0]["outcome_vector"])]
    failing = [key for key in keys if not all(groups[key][0]["outcome_vector"])]
    chosen_keys = []
    if passing:
        chosen_keys.append(passing[0])
    chosen_keys.extend(failing[:2])
    for key in keys:
        if key not in chosen_keys:
            chosen_keys.append(key)
    selected = []
    for key in chosen_keys:
        pool = groups[key]
        selected.append(pool[rng.randrange(len(pool))])
        if len(selected) >= count:
            break
    rng.shuffle(selected)
    return selected


def task_tokens(tokenizer, row, device):
    text = ("Coding task:\n" + row["task"] +
            "\nDevelop several plausible implementation hypotheses before committing.\n")
    batch = tokenizer(text, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
    return batch["input_ids"], batch["attention_mask"]


@torch.no_grad()
def teacher_targets(teacher, tokenizer, row, selected, device, center=True):
    texts = [f"Coding task:\n{row['task']}\nCandidate implementation:\n{item['source']}"
             for item in selected]
    batch = tokenizer(texts, padding=True, truncation=True, max_length=384,
                      return_tensors="pt").to(device)
    output = teacher(**batch, use_cache=False, output_hidden_states=True)
    positions = batch["attention_mask"].sum(dim=1) - 1
    hidden = output.hidden_states[-1][
        torch.arange(len(selected), device=device), positions]
    hidden = hidden.float()
    if center:
        hidden = hidden - hidden.mean(dim=0, keepdim=True)
    alive = torch.tensor([float(item["all_passed"]) for item in selected],
                         device=device)
    values = torch.tensor([item["value"] for item in selected], device=device)
    return hidden, alive, values


def assignment(predicted, targets):
    similarity = F.normalize(predicted.float(), dim=-1) @ F.normalize(
        targets.float(), dim=-1).T
    best_score, best = None, None
    for slots in itertools.permutations(range(predicted.shape[0]), targets.shape[0]):
        score = sum(float(similarity[slot, target])
                    for target, slot in enumerate(slots))
        if best_score is None or score > best_score:
            best_score, best = score, slots
    return torch.tensor(best, device=predicted.device), similarity


def family_pairs(selected):
    """Same-premise pairs (same error class and failing mask) versus pairs
    from different behavioral masks."""
    same, across = [], []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            a, b = selected[left], selected[right]
            if tuple(a["outcome_vector"]) == tuple(b["outcome_vector"]):
                continue
            family_a = (a["all_passed"], a["error_class"], a["failing_mask"])
            family_b = (b["all_passed"], b["error_class"], b["failing_mask"])
            if family_a == family_b:
                same.append((left, right))
            else:
                across.append((left, right))
    return same, across


def equivalence_pairs(selected):
    pairs = []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            if tuple(selected[left]["outcome_vector"]) == tuple(
                    selected[right]["outcome_vector"]):
                pairs.append((left, right))
    return pairs


def effective_rank(frontier):
    singular = torch.linalg.svdvals(frontier.float())
    probability = singular / singular.sum().clamp_min(1e-12)
    return float((-(probability * probability.clamp_min(1e-12).log()).sum()).exp())


def margin_loss(embeddings, same, across, margin=0.35):
    if not same or not across:
        return torch.zeros((), device=embeddings.device)
    normalized = F.normalize(embeddings, dim=-1)
    within = sum(F.relu(
        margin - (normalized[a] * normalized[b]).sum())
        for a, b in same) / len(same)
    between = sum(F.relu(
        margin + (normalized[a] * normalized[b]).sum())
        for a, b in across) / len(across)
    return within + between


@torch.no_grad()
def evaluate(model, teacher, tokenizer, rows, seed):
    model.eval(); rng = random.Random(seed)
    records, frontiers = [], []
    for row in rows:
        selected = select_candidates(row, rng)
        targets, alive, values = teacher_targets(
            teacher, tokenizer, row, selected, "cuda")
        ids, mask = task_tokens(tokenizer, row, "cuda")
        output = model(input_ids=ids, attention_mask=mask,
                       use_cache=False, coupling="zero")
        predicted = output.decoded_hypotheses[0].float()
        slots, similarity = assignment(predicted, targets)
        matched = similarity[slots, torch.arange(len(selected), device="cuda")]
        retrieval = similarity[slots].argmax(dim=-1)
        correct = int((retrieval == torch.arange(
            len(selected), device="cuda")).sum())
        alive_pred = output.slot_alive_logits[0, slots]
        value_pred = output.slot_values[0, slots]
        same, across = family_pairs(selected)
        family_margin = family_gap(predicted[slots], selected)
        records.append({
            "mean_cosine": float(matched.mean()),
            "retrieval_correct": correct, "retrieval_total": len(selected),
            "alive_correct": int(((alive_pred > 0) == alive.bool()).sum()),
            "value_mse": float(F.mse_loss(value_pred, values)),
            "family_margin": family_margin,
            "multiplicity_before": len(selected),
            "multiplicity_after": len({
                tuple(selected[int(c)]["outcome_vector"])
                for c in retrieval.tolist()}),
        })
        frontiers.append(output.frontier_state[0].float().cpu())
    baseline = torch.stack(frontiers).mean(dim=0)
    ranks = [effective_rank(frontier - baseline) for frontier in frontiers]
    total = sum(r["multiplicity_before"] for r in records)
    compressed = sum(r["multiplicity_after"] for r in records)
    return {
        "tasks": len(rows),
        "mean_matched_cosine": statistics.mean(r["mean_cosine"] for r in records),
        "candidate_decoding_accuracy": sum(r["retrieval_correct"] for r in records)
        / sum(r["retrieval_total"] for r in records),
        "alive_accuracy": sum(r["alive_correct"] for r in records)
        / sum(r["retrieval_total"] for r in records),
        "mean_value_mse": statistics.mean(r["value_mse"] for r in records),
        "mean_family_margin": statistics.mean(r["family_margin"] for r in records),
        "multiplicity_reduction": 1 - compressed / max(total, 1),
        "mean_task_conditioned_effective_rank": statistics.mean(ranks),
    }


def family_gap(embeddings, selected):
    """Mean cross-candidate cosine separated by same-family versus different
    behavioral masks, computed on retrieved slot vectors."""
    normalized = F.normalize(embeddings, dim=-1)
    within, across = [], []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            a, b = selected[left], selected[right]
            gap = float((normalized[left] * normalized[right]).sum())
            if tuple(a["outcome_vector"]) == tuple(b["outcome_vector"]):
                within.append(gap)
            else:
                family_a = (a["all_passed"], a["error_class"], a["failing_mask"])
                family_b = (b["all_passed"], b["error_class"], b["failing_mask"])
                (within if family_a == family_b else across).append(gap)
    mean_within = statistics.mean(within) if within else 0.0
    mean_across = statistics.mean(across) if across else 0.0
    return mean_within - mean_across


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/mbpp_v4_families/rows.jsonl")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--resume-native-state")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if len(json.loads(line)["candidates"]) >= 4]
    split = int(len(rows) * 0.8); train, validation = rows[:split], rows[split:]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=model_config(args.model, tokenizer), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.model.reset_native_output(); model.model.set_native_dtype(torch.float32)
    if args.resume_native_state:
        state = torch.load(args.resume_native_state, map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state, strict=False)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    native_named = [(name, parameter) for name, parameter in model.named_parameters()
                    if ("hypothesis_update" in name or "frontier_init" in name or
                        "hypothesis_decoder" in name or "slot_alive_head" in name or
                        "slot_value_head" in name)]
    native = [parameter for _, parameter in native_named]
    for parameter in native: parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(native, lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    before = evaluate(model, teacher, tokenizer, validation, args.seed + 1000)
    rng = random.Random(args.seed); losses, times = [], []
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for step in range(1, args.steps + 1):
        row = rng.choice(train); selected = select_candidates(row, rng)
        targets, alive, values = teacher_targets(
            teacher, tokenizer, row, selected, "cuda")
        ids, mask = task_tokens(tokenizer, row, "cuda")
        tick = time.perf_counter(); model.train()
        output = model(input_ids=ids, attention_mask=mask,
                       use_cache=False, coupling="zero")
        predicted = output.decoded_hypotheses[0].float()
        slots, similarity = assignment(predicted.detach(), targets)
        target_ids = torch.arange(len(selected), device="cuda")
        coverage = (1 - F.cosine_similarity(predicted[slots], targets, dim=-1)).mean()
        matched_logits = similarity[slots] / 0.07
        contrastive = F.cross_entropy(matched_logits, target_ids)
        alive_logits = output.slot_alive_logits[0]
        alive_targets = torch.zeros_like(alive_logits); alive_targets[slots] = alive
        alive_loss = F.binary_cross_entropy_with_logits(alive_logits, alive_targets)
        value_targets = torch.zeros_like(output.slot_values[0]); value_targets[slots] = values
        value_loss = F.mse_loss(output.slot_values[0], value_targets)
        same, across = family_pairs(selected)
        family = margin_loss(predicted[slots], same, across)
        collapse = torch.zeros((), device="cuda")
        eq = equivalence_pairs(selected)
        if eq:
            normalized = F.normalize(predicted[slots], dim=-1)
            collapse = sum(1 - (normalized[a] * normalized[b]).sum()
                           for a, b in eq) / len(eq)
        ownership = similarity.softmax(dim=0)
        overlap = 0.0; pairs = 0
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                overlap = overlap + (ownership[:, left] * ownership[:, right]).sum()
                pairs += 1
        separation = overlap / max(pairs, 1)
        loss = (coverage + 0.5 * contrastive + 0.2 * alive_loss
                + 0.1 * value_loss + 0.1 * family + 0.1 * collapse
                + 0.02 * separation)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(native, 1.0); optimizer.step()
        torch.cuda.synchronize(); losses.append(float(loss.detach()))
        times.append(time.perf_counter() - tick)
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": round(losses[-1], 5),
                              "coverage": round(float(coverage.detach()), 5),
                              "family": round(float(family.detach()), 5),
                              "collapse": round(float(collapse.detach()), 5),
                              "elapsed_seconds": round(
                                  time.perf_counter() - started, 2)}),
                  flush=True)
    after = evaluate(model, teacher, tokenizer, validation, args.seed + 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "native_interference_state.pt"
    torch.save({name: parameter.detach().cpu() for name, parameter in native_named},
               state_path)
    report = {"status": "V4a family hypothesis-set admission",
              "source_commit": args.source_commit,
              "train_tasks": len(train), "validation_tasks": len(validation),
              "steps": args.steps, "before": before, "after": after,
              "resume_native_state": args.resume_native_state,
              "loss_first": losses[0], "loss_last": losses[-1],
              "median_step_seconds": statistics.median(times),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
              "native_state_path": str(state_path)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
