"""One-hour native Interference Search gate on real SWE trajectories."""

import argparse
import hashlib
import importlib
import json
import random
import statistics
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

from unsloth import FastLanguageModel
import torch
from torch.nn import functional as F

from interference_search.native_kv_frontier import NativeKVFrontier


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def tokens(tokenizer, texts, length):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=length,
                      add_special_tokens=False, return_tensors="pt")
    return batch["input_ids"].cuda(), batch["attention_mask"].cuda().bool()


def restore_reference_qwen3(backbone):
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    qwen3 = importlib.reload(qwen3)
    backbone.model.rotary_emb = qwen3.Qwen3RotaryEmbedding(
        config=backbone.config).to(device="cuda")
    backbone.forward = types.MethodType(qwen3.Qwen3ForCausalLM.forward, backbone)
    backbone.model.forward = types.MethodType(qwen3.Qwen3Model.forward, backbone.model)
    for layer in backbone.model.layers:
        layer.forward = types.MethodType(qwen3.Qwen3DecoderLayer.forward, layer)
        layer.self_attn.forward = types.MethodType(
            qwen3.Qwen3Attention.forward, layer.self_attn)


def encode_row(model, tokenizer, row):
    task_ids, task_mask = tokens(tokenizer, [row["task"]], 384)
    context = model.prepare(task_ids, task_mask)
    branches = row["branches"]
    failed = [index for index, branch in enumerate(branches) if not branch["resolved"]]
    suffixes = [branch["state"] for branch in branches]
    suffixes += [branch["action"] for branch in branches]
    suffixes += [branch["observation"] for branch in branches]
    suffixes += [branches[index]["observation"] for index in failed]
    ids, mask = tokens(tokenizer, suffixes, 256)
    vectors = model.encode_suffixes(context, ids, mask)
    width = len(branches)
    states = vectors[:width].unsqueeze(0)
    actions = vectors[width:2 * width][None, :, None]
    observations = vectors[2 * width:3 * width]
    refs = vectors[3 * width:].unsqueeze(0)
    labels = torch.tensor([branch["resolved"] for branch in branches],
                          dtype=torch.float32, device="cuda")
    positive = torch.zeros(1, width, 1, len(failed), device="cuda")
    for ref_index, branch_index in enumerate(failed):
        positive[0, branch_index, 0, ref_index] = 1
    return context, states, actions, observations, refs, labels, positive


def forward_row(model, tokenizer, row, coupling):
    context, states, actions, observations, refs, labels, positive = encode_row(
        model, tokenizer, row)
    output = model.frontier_step(
        context, states, torch.ones(states.shape[:2], dtype=torch.bool, device="cuda"),
        refs, torch.ones(refs.shape[:2], dtype=torch.bool, device="cuda"),
        actions, torch.ones(1, states.shape[1], 1, dtype=torch.bool, device="cuda"),
        coupling)
    score = output.survival_logits[0] + output.proposal_logits[0, :, 0]
    return output, score, observations, labels, positive


def row_loss(model, tokenizer, row, coupling):
    output, score, observations, labels, positive = forward_row(
        model, tokenizer, row, coupling)
    live = labels.bool()
    good, bad = score[live], score[~live]
    rank = F.softplus(bad[None, :] - good[:, None]).mean()
    classification = F.binary_cross_entropy_with_logits(score, labels)
    predicted = output.predicted_successors[0, :, 0]
    transition = 1 - (F.normalize(predicted, dim=-1) *
                      F.normalize(observations.detach().float(), dim=-1)).sum(dim=-1).mean()
    logits = output.action_refutation_logits
    target = positive
    positives = target.sum().clamp_min(1)
    negatives = (target.numel() - positives).clamp_min(1)
    match = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=(negatives / positives).detach())
    loss = rank + 0.3 * classification + 0.3 * transition + 0.3 * match
    return loss, {"rank": rank, "classification": classification,
                  "transition": transition, "match": match}


@torch.no_grad()
def evaluate(model, tokenizer, rows):
    model.eval()
    result = {}
    for coupling in ("zero", "inhibit"):
        top1 = pair_correct = pairs = 0
        positive_matches, negative_matches = [], []
        for row in rows:
            output, score, _, labels, positive = forward_row(
                model, tokenizer, row, coupling)
            top1 += int(labels[int(score.argmax())].item())
            good = score[labels.bool()]
            bad = score[~labels.bool()]
            pair_correct += int((good[:, None] > bad[None, :]).sum())
            pairs += good.numel() * bad.numel()
            matches = torch.sigmoid(output.action_refutation_logits)
            positive_matches.append(float(matches[positive.bool()].mean()))
            negative_matches.append(float(matches[~positive.bool()].mean()))
        pos = statistics.mean(positive_matches)
        neg = statistics.mean(negative_matches)
        result[coupling] = {
            "rows": len(rows), "top1_resolved_rate": top1 / len(rows),
            "resolved_over_failed_pair_accuracy": pair_correct / max(pairs, 1),
            "refutation_match_positive": pos,
            "refutation_match_negative": neg,
            "refutation_match_gap": pos - neg,
        }
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_order(rows, count, seed):
    """Sample repositories uniformly, then sample a frontier inside the repo."""
    by_repo = defaultdict(list)
    for index, row in enumerate(rows):
        by_repo[row["repo_family"]].append(index)
    rng = random.Random(seed)
    repos = sorted(by_repo)
    return [rng.choice(by_repo[rng.choice(repos)]) for _ in range(count)]


def evaluation_sample(rows, count, seed):
    if count <= 0 or count >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/real_swe_frontier_v0"))
    parser.add_argument("--out", type=Path, default=Path("runs/real_swe_frontier_gate"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-rows", type=int, default=80)
    parser.add_argument("--head-warmup", type=int, default=25)
    parser.add_argument("--inhibit-start", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--match-space", choices=("learned", "model"), default="learned")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train = read_rows(args.data / "train.jsonl")
    validation = evaluation_sample(
        read_rows(args.data / "validation.jsonl"), args.eval_rows, args.seed + 1)
    test = evaluation_sample(
        read_rows(args.data / "test.jsonl"), args.eval_rows, args.seed + 2)
    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path, max_seq_length=768, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=False)
    peft_model = FastLanguageModel.get_peft_model(
        peft_model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=args.seed, use_rslora=True)
    backbone = peft_model.get_base_model()
    restore_reference_qwen3(backbone)
    model = NativeKVFrontier(backbone, actions=1, train_backbone=True,
                             scale=args.scale, match_space=args.match_space).cuda()
    before = {"validation": evaluate(model, tokenizer, validation),
              "test": evaluate(model, tokenizer, test)}
    head = list(model.cell.parameters())
    lora = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    head_before = model.cell.value_head.weight.detach().cpu().clone()
    lora_before = lora[0].detach().cpu().clone()
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": args.head_lr},
        {"params": lora, "lr": 0.0},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    order = balanced_order(train, args.steps, args.seed)
    losses, times = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, index in enumerate(order, 1):
        model.train()
        if step == args.head_warmup + 1:
            optimizer.param_groups[1]["lr"] = args.lora_lr
        coupling = "zero" if step <= args.inhibit_start else "inhibit"
        if (step - 1) % args.grad_accum == 0:
            optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        loss, parts = row_loss(model, tokenizer, train[index], coupling)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        times.append(time.perf_counter() - tick)
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps({"step": step, "coupling": coupling,
                              "loss": round(losses[-1], 5),
                              **{key: round(float(value.detach()), 5)
                                 for key, value in parts.items()},
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    after = {"validation": evaluate(model, tokenizer, validation),
             "test": evaluate(model, tokenizer, test)}
    args.out.mkdir(parents=True, exist_ok=True)
    head_path = args.out / "native_frontier_head.pt"
    adapter_path = args.out / "unsloth_adapter"
    torch.save(model.cell.state_dict(), head_path)
    peft_model.save_pretrained(adapter_path, safe_serialization=True)
    report = {
        "status": "real SWE native frontier one-hour gate",
        "model_path": args.model_path, "source_commit": args.source_commit,
        "steps": args.steps, "seed": args.seed,
        "training_config": {
            "sampling": "uniform_repo_then_uniform_row",
            "match_space": args.match_space,
            "scale": args.scale,
            "grad_accum": args.grad_accum,
            "head_lr": args.head_lr,
            "lora_lr": args.lora_lr,
            "head_warmup": args.head_warmup,
            "inhibit_start": args.inhibit_start,
        },
        "data_sha256": {split: sha256(args.data / f"{split}.jsonl")
                        for split in ("train", "validation", "test")},
        "before": before, "after": after,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_last_25_mean": statistics.mean(losses[-25:]),
        "training_seconds": time.perf_counter() - started,
        "steady_step_seconds": statistics.median(times[1:]),
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "head_changed": not torch.equal(head_before, model.cell.value_head.weight.detach().cpu()),
        "lora_changed": not torch.equal(lora_before, lora[0].detach().cpu()),
        "trainable_head_parameters": sum(p.numel() for p in head),
        "trainable_lora_parameters": sum(p.numel() for p in lora),
        "head_path": str(head_path), "adapter_path": str(adapter_path),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
