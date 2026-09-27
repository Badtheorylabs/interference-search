"""Midtrain native Interference Qwen3 on verified same-state code actions."""

import argparse
import hashlib
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


def clipped(tokenizer, text, limit, tail=False):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[-limit:] if tail else ids[:limit]


def example(tokenizer, row, rng, device):
    negative = rng.choice(row["negatives"])
    positive = row["positive"]
    task = clipped(tokenizer, "Coding task:\n" + row["task"] + "\n", 128)
    positive_first = rng.random() < 0.5
    source_a = positive["source"] if positive_first else negative["source"]
    source_b = negative["source"] if positive_first else positive["source"]
    candidate_a = clipped(tokenizer, "Candidate A:\n" + source_a + "\n", 192)
    candidate_b = clipped(tokenizer, "Candidate B:\n" + source_b + "\n", 192)
    feedback_text = "\n".join(
        f"{status}: {detail}" for status, detail in negative["tests"])
    feedback = clipped(
        tokenizer, "Verified execution failure:\n" + feedback_text + "\n", 128)
    instruction = clipped(
        tokenizer,
        "The verified failure applies to the failing candidate. "
        "Which candidate is the correct implementation? Answer A or B:\n", 40)
    marker = tokenizer.encode("<tool_response>", add_special_tokens=False)
    if len(marker) != 1:
        raise ValueError("Qwen tool response marker must be one token")
    if positive_first:
        ids = task + candidate_a + candidate_b + feedback + marker + instruction
        answer = "A"
    else:
        ids = task + candidate_a + feedback + marker + candidate_b + instruction
        answer = "B"
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not target_ids:
        return None
    return (torch.tensor(ids, device=device)[None],
            torch.ones(1, len(ids), dtype=torch.long, device=device),
            torch.tensor([target_ids[0]], device=device))


@torch.no_grad()
def evaluate(model, teacher, tokenizer, rows, seed):
    model.eval()
    rng = random.Random(seed)
    inhibit_values, zero_values, margins, kls = [], [], [], []
    for row in rows:
        ids, mask, target = example(tokenizer, row, rng, "cuda")
        teacher_logits = teacher(input_ids=ids, attention_mask=mask,
                                 use_cache=False).logits[:, -1].float()
        zero = model(input_ids=ids, attention_mask=mask, use_cache=False,
                     coupling="zero").logits[:, -1].float()
        inhibit = model(input_ids=ids, attention_mask=mask, use_cache=False,
                        coupling="inhibit").logits[:, -1].float()
        zero_lp = zero.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        inhibit_lp = inhibit.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        zero_values.append(float(zero_lp))
        inhibit_values.append(float(inhibit_lp))
        margins.append(float(inhibit_lp - zero_lp))
        kls.append(float(F.kl_div(
            zero.log_softmax(dim=-1), teacher_logits.softmax(dim=-1),
            reduction="batchmean")))
    return {"rows": len(rows),
            "mean_inhibit_target_logp": statistics.mean(inhibit_values),
            "mean_zero_target_logp": statistics.mean(zero_values),
            "mean_inhibit_minus_zero": statistics.mean(margins),
            "positive_margin_rows": sum(value > 0 for value in margins),
            "mean_zero_vs_teacher_kl": statistics.mean(kls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/mbpp_same_state_train/rows.jsonl")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    split = max(1, int(len(rows) * 0.8))
    train, validation = rows[:split], rows[split:]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    values = base_config.to_dict()
    values.pop("model_type", None)
    values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_response>"))
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    model.model.reset_native_output()
    model.model.set_native_dtype(torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    native_named = [(name, parameter) for name, parameter in model.named_parameters()
                    if "hypothesis_update" in name or "frontier_init" in name]
    native = [parameter for _, parameter in native_named]
    for parameter in native:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        native, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    before = evaluate(model, teacher, tokenizer, validation, args.seed + 1000)
    rng = random.Random(args.seed)
    losses, margins, times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids, mask, target = example(
            tokenizer, rng.choice(train), rng, "cuda")
        tick = time.perf_counter()
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=ids, attention_mask=mask,
                use_cache=False).logits[:, -1].float()
        model.train()
        zero = model(input_ids=ids, attention_mask=mask, use_cache=False,
                     coupling="zero").logits[:, -1].float()
        inhibit = model(input_ids=ids, attention_mask=mask, use_cache=False,
                        coupling="inhibit")
        inhibit_logits = inhibit.logits[:, -1].float()
        zero_lp = zero.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        inhibit_lp = inhibit_logits.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        margin = inhibit_lp - zero_lp
        preserve = F.kl_div(
            zero.log_softmax(dim=-1), teacher_logits.softmax(dim=-1),
            reduction="batchmean")
        slots = F.normalize(inhibit.frontier_state.float(), dim=-1)
        cosine = torch.matmul(slots, slots.transpose(1, 2))
        eye = torch.eye(cosine.shape[-1], dtype=torch.bool, device=cosine.device)
        diversity = cosine[:, ~eye].square().mean()
        variance = F.relu(0.1 - inhibit.frontier_state.float().std(dim=1)).mean()
        loss = (-inhibit_lp + F.relu(0.05 - margin)
                + args.preserve_weight * preserve
                + args.diversity_weight * (diversity + variance))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(native, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        margins.append(float(margin.detach()))
        times.append(time.perf_counter() - tick)
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": round(losses[-1], 5),
                              "causal_margin": round(margins[-1], 6),
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}),
                  flush=True)
    after = evaluate(model, teacher, tokenizer, validation, args.seed + 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "native_interference_state.pt"
    torch.save({name: parameter.detach().cpu() for name, parameter in native_named},
               state_path)
    report = {
        "status": "same-state verified coding midtrain",
        "source_commit": args.source_commit,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_tasks": len(train), "validation_tasks": len(validation),
        "preserve_weight": args.preserve_weight,
        "diversity_weight": args.diversity_weight,
        "steps": args.steps, "before": before, "after": after,
        "loss_first": losses[0], "loss_last": losses[-1],
        "mean_training_margin": statistics.mean(margins),
        "median_step_seconds": statistics.median(times),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "native_state_path": str(state_path),
        "native_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
