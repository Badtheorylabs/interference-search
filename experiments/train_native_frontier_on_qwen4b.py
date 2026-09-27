"""Train Interference Search natively on Qwen3-4B hidden states and task KV.

The frozen backbone prefills each task once. Frontier states, verified failures,
typed actions, and executed successors pass through the same model weights.
State branches reuse the task KV. This stage updates the native search and
transition heads while holding the 4B backbone fixed.
"""

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
from pathlib import Path

import torch
from torch.nn import functional as F

if "--unsloth" in sys.argv:
    from unsloth import FastLanguageModel
else:
    FastLanguageModel = None

from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.countdown import solver
from interference_search.countdown_actions import CountdownActionCodec
from interference_search.native_kv_frontier import NativeKVFrontier


MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CODEC = CountdownActionCodec(max_numbers=7)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_text(values):
    return "remaining values: " + ", ".join(map(str, values))


def task_text(row):
    return (f"Use each number {row['numbers']} once with + - * / to reach "
            f"{row['target']}. Intermediate values must be positive whole numbers.")


def action_text(action):
    word = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[action.op]
    return f"{word} sorted position {action.left} with sorted position {action.right}"


def tokens(tokenizer, texts, length):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=length,
                        return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool()


def encode_row(model, tokenizer, row):
    task_ids, task_mask = tokens(tokenizer, [task_text(row)], 96)
    context = model.prepare(task_ids, task_mask)
    suffixes = [state_text(state) for state in row["frontier"]]
    suffixes.append(state_text(row["verified_dead_state"]))
    successor_specs = []
    for state_index, records in enumerate(row["sampled_successors"]):
        for action_id, successor in records:
            successor_specs.append((state_index, action_id))
            suffixes.append(state_text(successor))
    suffix_ids, suffix_mask = tokens(tokenizer, suffixes, 32)
    vectors = model.encode_suffixes(context, suffix_ids, suffix_mask)
    width = len(row["frontier"])
    states = vectors[:width].unsqueeze(0)
    refs = vectors[width:width + 1].unsqueeze(0)
    successors = vectors[width + 1:]
    legal = torch.tensor([CODEC.legal_mask(tuple(state)) for state in row["frontier"]],
                         dtype=torch.bool, device="cuda").unsqueeze(0)
    return context, states, refs, successors, successor_specs, legal


def forward_row(model, tokenizer, row, action_vectors, coupling):
    context, states, refs, successors, successor_specs, legal = encode_row(
        model, tokenizer, row)
    output = model.frontier_step(
        context, states, torch.ones(states.shape[:2], dtype=torch.bool, device="cuda"),
        refs, torch.ones(refs.shape[:2], dtype=torch.bool, device="cuda"),
        action_vectors, legal, coupling)
    return output, successors, successor_specs, legal


def row_loss(model, tokenizer, row, action_vectors, coupling):
    output, successors, specs, legal = forward_row(
        model, tokenizer, row, action_vectors, coupling)
    can_solve = solver(row["target"])
    action_losses = []
    for state_index, values in enumerate(row["frontier"]):
        state = tuple(values)
        legal_ids = [index for index, allowed in enumerate(CODEC.legal_mask(state))
                     if allowed]
        winning_ids = [index for index in legal_ids
                       if can_solve(CODEC.apply(state, index))]
        if winning_ids:
            logits = output.proposal_logits[0, state_index]
            action_losses.append(torch.logsumexp(logits[legal_ids], dim=0) -
                                 torch.logsumexp(logits[winning_ids], dim=0))
    action_loss = torch.stack(action_losses).mean()
    survival = torch.tensor(row["survival_targets"], dtype=torch.float32, device="cuda")
    survival_loss = F.binary_cross_entropy_with_logits(output.survival_logits[0], survival)
    refuted = torch.zeros_like(output.proposal_logits)
    for state_index, actions in enumerate(row["refutation_actions"]):
        refuted[0, state_index, actions] = 1
    match_logits = output.action_refutation_logits.squeeze(-1)[legal]
    match_targets = refuted[legal]
    positive = match_targets.sum().clamp_min(1)
    negative = (match_targets.numel() - positive).clamp_min(1)
    match_loss = F.binary_cross_entropy_with_logits(
        match_logits, match_targets, pos_weight=(negative / positive).detach())
    predicted = torch.stack([output.predicted_successors[0, state_index, action_id]
                             for state_index, action_id in specs])
    transition_loss = 1 - (F.normalize(predicted, dim=-1) *
                           F.normalize(successors.detach().float(), dim=-1)).sum(dim=-1).mean()
    loss = action_loss + 0.4 * survival_loss + 0.5 * match_loss + 0.3 * transition_loss
    return loss, {"action": action_loss, "survival": survival_loss,
                  "match": match_loss, "transition": transition_loss}


@torch.no_grad()
def evaluate(model, tokenizer, rows, action_vectors):
    model.eval()
    metrics = {mode: {"exact": 0, "winning": 0, "alive": 0,
                      "refuted": 0, "opportunities": 0,
                      "positive": [], "negative": []}
               for mode in ("zero", "inhibit")}
    for row in rows:
        context, states, refs, _, _, legal = encode_row(model, tokenizer, row)
        target_solver = solver(row["target"])
        for mode, values in metrics.items():
            output = model.frontier_step(
                context, states, torch.ones(states.shape[:2], dtype=torch.bool, device="cuda"),
                refs, torch.ones(refs.shape[:2], dtype=torch.bool, device="cuda"),
                action_vectors, legal, mode)
            chosen = output.proposal_logits[0].argmax(dim=-1).cpu()
            refuted_mask = torch.zeros_like(output.proposal_logits, dtype=torch.bool)
            for index, state_values in enumerate(row["frontier"]):
                target = row["action_targets"][index]
                action_id = int(chosen[index])
                if target >= 0:
                    values["alive"] += 1
                    values["exact"] += int(action_id == target)
                    values["winning"] += int(
                        target_solver(CODEC.apply(tuple(state_values), action_id)))
                refuted = set(row["refutation_actions"][index])
                refuted_mask[0, index, list(refuted)] = True
                if refuted:
                    values["opportunities"] += 1
                    values["refuted"] += int(action_id in refuted)
            positive = refuted_mask & legal
            negative = ~refuted_mask & legal
            values["positive"].append(float(output.action_refutation_match[positive].mean()))
            values["negative"].append(float(output.action_refutation_match[negative].mean()))
    result = {}
    for mode, values in metrics.items():
        positive = statistics.mean(values["positive"])
        negative = statistics.mean(values["negative"])
        result[mode] = {
            "rows": len(rows), "alive_states": values["alive"],
            "exact_target_top1_rate": values["exact"] / max(values["alive"], 1),
            "winning_action_top1_rate": values["winning"] / max(values["alive"], 1),
            "known_refuted_action_top1_rate": values["refuted"] /
                                                max(values["opportunities"], 1),
            "refutation_match_positive": positive,
            "refutation_match_negative": negative,
            "refutation_match_gap": positive - negative,
        }
    return result


def source_receipt(source_commit):
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=root, text=True).strip())
    except subprocess.CalledProcessError:
        if not source_commit:
            raise RuntimeError("copied source requires --source-commit")
        commit, dirty = source_commit, False
    return commit, dirty, sha256(Path(__file__))


def restore_reference_qwen3_forward(backbone):
    """Keep Unsloth LoRA modules but use differentiable HF cache semantics."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/native_countdown"))
    parser.add_argument("--out", type=Path, default=Path("runs/qwen4b_native_kv"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-rows", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--inhibit-start", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-commit")
    parser.add_argument("--unsloth", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-warmup", type=int, default=50)
    args = parser.parse_args()
    commit, dirty, script_hash = source_receipt(args.source_commit)
    if dirty:
        raise RuntimeError("commit the exact training source before starting")
    paths = {name: args.data / f"{name}.jsonl"
             for name in ("train", "validation", "test_5_numbers")}
    rows = {name: read_rows(path) for name, path in paths.items()}
    peft_model = None
    if args.unsloth:
        model_path = args.model_path or MODEL
        peft_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=256, dtype=torch.bfloat16,
            load_in_4bit=False, full_finetuning=False,
        )
        peft_model = FastLanguageModel.get_peft_model(
            peft_model, r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_rank * 2, lora_dropout=0, bias="none",
            use_gradient_checkpointing=False, random_state=args.seed,
            use_rslora=True,
        )
        backbone = peft_model.get_base_model()
        restore_reference_qwen3_forward(backbone)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                  local_files_only=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="cuda").eval()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = NativeKVFrontier(backbone, len(CODEC.actions),
                             train_backbone=args.unsloth,
                             match_space="model").cuda()
    model.eval()
    action_ids, action_mask = tokens(tokenizer,
                                     [action_text(action) for action in CODEC.actions], 24)
    action_vectors = model.encode_standalone(action_ids, action_mask).detach()
    initial = {name: value.detach().cpu().clone()
               for name, value in model.cell.state_dict().items()}
    validation = rows["validation"][:args.eval_rows]
    test = rows["test_5_numbers"][:args.eval_rows]
    before = {"validation": evaluate(model, tokenizer, validation, action_vectors),
              "test_5_numbers": evaluate(model, tokenizer, test, action_vectors)}
    head_parameters = list(model.cell.parameters())
    lora_parameters = [parameter for parameter in backbone.parameters()
                       if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.learning_rate},
        {"params": lora_parameters, "lr": 0.0 if args.unsloth else args.lora_learning_rate},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    lora_before = [parameter.detach().cpu().clone() for parameter in lora_parameters[:2]]
    order = random.Random(args.seed).choices(range(len(rows["train"])), k=args.steps)
    losses, step_times = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, index in enumerate(order, 1):
        coupling = "zero" if step <= args.inhibit_start else "inhibit"
        model.train()
        # Qwen and the injected LoRA adapters use zero dropout in this recipe.
        if args.unsloth and step == args.head_warmup + 1:
            optimizer.param_groups[1]["lr"] = args.lora_learning_rate
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        loss, parts = row_loss(model, tokenizer, rows["train"][index],
                               action_vectors, coupling)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cell.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        step_times.append(time.perf_counter() - tick)
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps({"step": step, "coupling": coupling,
                              "loss": round(losses[-1], 5),
                              **{name: round(float(value.detach()), 5)
                                 for name, value in parts.items()},
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    after = {"validation": evaluate(model, tokenizer, validation, action_vectors),
             "test_5_numbers": evaluate(model, tokenizer, test, action_vectors)}
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "native_kv_frontier_head.pt"
    torch.save(model.cell.state_dict(), checkpoint)
    action_checkpoint = args.out / "native_action_vectors.pt"
    torch.save(action_vectors.detach().cpu(), action_checkpoint)
    adapter_dir = None
    if peft_model is not None:
        adapter_dir = args.out / "unsloth_adapter"
        peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    reloaded = NativeKVFrontier(backbone, len(CODEC.actions)).cuda()
    reloaded.cell.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    reload_reference = evaluate(model, tokenizer, validation[:4], action_vectors)
    if reload_reference != evaluate(
            reloaded, tokenizer, validation[:4], action_vectors):
        raise RuntimeError("save/reload behavior changed")
    adapter_reload_equal = None
    if adapter_dir is not None:
        loaded_peft, loaded_tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir), max_seq_length=256,
            dtype=torch.bfloat16, load_in_4bit=False)
        loaded_backbone = loaded_peft.get_base_model()
        restore_reference_qwen3_forward(loaded_backbone)
        loaded_native = NativeKVFrontier(
            loaded_backbone, len(CODEC.actions), train_backbone=True,
            match_space="model").cuda().eval()
        loaded_native.cell.load_state_dict(torch.load(
            checkpoint, map_location="cpu", weights_only=True))
        loaded_actions = torch.load(action_checkpoint, map_location="cuda",
                                    weights_only=True)
        adapter_reload_equal = reload_reference == evaluate(
            loaded_native, loaded_tokenizer, validation[:4], loaded_actions)
        if not adapter_reload_equal:
            raise RuntimeError("Unsloth adapter reload behavior changed")
    changed = sum(not torch.equal(initial[name], value.cpu())
                  for name, value in model.cell.state_dict().items())
    report = {
        "status": "trained native frontier on Qwen3-4B hidden states and live task KV",
        "claim_boundary": "native-head stage; Qwen backbone frozen; Countdown generator only",
        "model": MODEL, "revision": REVISION, "git_commit": commit,
        "training_script_sha256": script_hash,
        "data_sha256": {name: sha256(path) for name, path in paths.items()},
        "steps": args.steps, "learning_rate": args.learning_rate,
        "inhibit_start": args.inhibit_start, "seed": args.seed,
        "unsloth": args.unsloth, "lora_rank": args.lora_rank if args.unsloth else None,
        "backbone_forward": ("unsloth_lora_with_reference_qwen3_kv"
                             if args.unsloth else "transformers_qwen3"),
        "lora_learning_rate": args.lora_learning_rate if args.unsloth else None,
        "head_warmup": args.head_warmup if args.unsloth else None,
        "before": before, "after": after,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_last_25_mean": statistics.mean(losses[-25:]),
        "training_seconds": time.perf_counter() - started,
        "steady_step_seconds": statistics.median(step_times[1:]),
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "trainable_parameters": sum(parameter.numel() for parameter in model.cell.parameters()),
        "backbone_trainable_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "lora_changed": any(not torch.equal(before, after.detach().cpu())
                            for before, after in zip(lora_before, lora_parameters[:2])),
        "changed_head_tensors": changed,
        "task_prefills": model.task_prefills, "suffix_forwards": model.suffix_forwards,
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "action_vectors": str(action_checkpoint),
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "save_reload_behavior_equal": True,
        "adapter_reload_behavior_equal": adapter_reload_equal,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "before": before,
                      "after": after, "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
