"""Native-parameter midtraining admission on real SWE agent trajectories."""

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


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def choose_pair(row):
    good = next((branch for branch in row["branches"] if branch["resolved"]), None)
    bad = next((branch for branch in row["branches"] if not branch["resolved"]), None)
    return good, bad


def clipped_ids(tokenizer, text, limit, tail=False):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[-limit:] if tail else ids[:limit]


def training_example(tokenizer, row, rng, device):
    good, bad = choose_pair(row)
    if good is None or bad is None:
        return None
    task = clipped_ids(tokenizer, row["task"], 240)
    state = clipped_ids(tokenizer, bad["state"], 240, tail=True)
    observation = clipped_ids(tokenizer, bad["observation"], 128, tail=True)
    instruction = clipped_ids(
        tokenizer, "\nVerified tool failure above. Continue with the corrected next tool action:\n", 32)
    action = clipped_ids(tokenizer, good["action"], 96)
    if not action:
        return None
    cut = rng.randrange(len(action))
    prefix = task + state
    observation_start = len(prefix)
    ids = prefix + observation + instruction + action[:cut]
    target = action[cut]
    refutation = [False] * len(ids)
    for index in range(observation_start, observation_start + len(observation)):
        refutation[index] = True
    return (torch.tensor(ids, device=device)[None],
            torch.ones(1, len(ids), dtype=torch.long, device=device),
            torch.tensor(refutation, dtype=torch.bool, device=device)[None],
            torch.tensor([target], device=device))


@torch.no_grad()
def evaluate(model, teacher, tokenizer, rows, count, seed):
    model.eval()
    rng = random.Random(seed)
    margins, target_logps, zero_logps, kls = [], [], [], []
    used = 0
    for row in rows:
        example = training_example(tokenizer, row, rng, "cuda")
        if example is None:
            continue
        ids, mask, refutation, target = example
        teacher_logits = teacher(input_ids=ids, attention_mask=mask,
                                 use_cache=False).logits[:, -1].float()
        zero = model(input_ids=ids, attention_mask=mask, use_cache=False,
                     coupling="zero", refutation_mask=refutation).logits[:, -1].float()
        inhibit = model(input_ids=ids, attention_mask=mask, use_cache=False,
                        coupling="inhibit", refutation_mask=refutation).logits[:, -1].float()
        teacher_prob = teacher_logits.softmax(dim=-1)
        kls.append(float(F.kl_div(zero.log_softmax(dim=-1), teacher_prob,
                                 reduction="batchmean")))
        zero_logp = zero.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        inhibit_logp = inhibit.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        zero_logps.append(float(zero_logp))
        target_logps.append(float(inhibit_logp))
        margins.append(float(inhibit_logp - zero_logp))
        used += 1
        if used >= count:
            break
    return {"rows": used,
            "mean_inhibit_target_logp": statistics.mean(target_logps),
            "mean_zero_target_logp": statistics.mean(zero_logps),
            "mean_inhibit_minus_zero": statistics.mean(margins),
            "mean_zero_vs_teacher_kl": statistics.mean(kls)}


def native_state(model):
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if "hypothesis_update" in name or "frontier_init" in name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/real_swe_frontier_v0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-rows", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train = read_rows(Path(args.data) / "train.jsonl")
    validation = read_rows(Path(args.data) / "validation.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    values = base_config.to_dict()
    values.pop("model_type", None)
    values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4)
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    model.model.reset_native_output()
    model.model.set_native_dtype(torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    native = [parameter for name, parameter in model.named_parameters()
              if "hypothesis_update" in name or "frontier_init" in name]
    for parameter in native:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(native, lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    before = evaluate(model, teacher, tokenizer, validation,
                      args.eval_rows, args.seed + 1000)
    rng = random.Random(args.seed)
    order = rng.choices(range(len(train)), k=args.steps)
    losses, margins, times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, index in enumerate(order, 1):
        example = training_example(tokenizer, train[index], rng, "cuda")
        if example is None:
            continue
        ids, mask, refutation, target = example
        tick = time.perf_counter()
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=ids, attention_mask=mask,
                use_cache=False).logits[:, -1].float()
        model.train()
        zero = model(input_ids=ids, attention_mask=mask, use_cache=False,
                     coupling="zero", refutation_mask=refutation).logits[:, -1].float()
        inhibit = model(input_ids=ids, attention_mask=mask, use_cache=False,
                        coupling="inhibit", refutation_mask=refutation).logits[:, -1].float()
        zero_logp = zero.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        inhibit_logp = inhibit.log_softmax(dim=-1).gather(1, target[:, None]).mean()
        task_loss = -inhibit_logp
        causal_margin = inhibit_logp - zero_logp
        margin_loss = F.relu(0.05 - causal_margin)
        preserve = F.kl_div(
            zero.log_softmax(dim=-1), teacher_logits.softmax(dim=-1),
            reduction="batchmean")
        loss = task_loss + 0.5 * margin_loss + 0.2 * preserve
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(native, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        margins.append(float(causal_margin.detach()))
        times.append(time.perf_counter() - tick)
        print(json.dumps({"step": step, "loss": round(losses[-1], 5),
                          "causal_margin": round(margins[-1], 6),
                          "elapsed_seconds": round(time.perf_counter() - started, 2)}),
              flush=True)
    after = evaluate(model, teacher, tokenizer, validation,
                     args.eval_rows, args.seed + 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "native_interference_state.pt"
    torch.save(native_state(model), state_path)
    report = {
        "status": "native Interference Qwen3 coding midtrain admission",
        "source_commit": args.source_commit, "steps": len(losses),
        "trainable_native_parameters": sum(p.numel() for p in native),
        "before": before, "after": after,
        "loss_first": losses[0], "loss_last": losses[-1],
        "mean_training_causal_margin": statistics.mean(margins),
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
