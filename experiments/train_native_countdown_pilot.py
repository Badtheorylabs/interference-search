"""Bounded Qwen3-4B native-frontier mechanism pilot on exact Countdown rows.

This tests whether a model learns action ranking, successor prediction, and
selective response to one verified cross-branch dead state. It does not test
open-ended language reasoning or establish a compute advantage.
"""

import argparse
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.countdown import solver
from interference_search.countdown_actions import CountdownActionCodec
from interference_search.native_lm import NativeFrontierLM


MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CODEC = CountdownActionCodec(max_numbers=7)


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(tokenizer, texts, length):
    result = tokenizer(texts, max_length=length, padding="max_length", truncation=True,
                       return_tensors="pt", add_special_tokens=False)
    return result["input_ids"], result["attention_mask"].bool()


def state_text(values):
    return "remaining numbers: " + ", ".join(map(str, values))


def pack_row(row, tokenizer):
    prompt = (f"Use all numbers {row['numbers']} once with + - * / to reach {row['target']}. "
              "Each intermediate result must be a positive whole number.")
    task_ids, task_mask = tokenize(tokenizer, [prompt], 96)
    state_ids, state_tokens = tokenize(tokenizer, [state_text(s) for s in row["frontier"]], 32)
    ref_ids, ref_tokens = tokenize(tokenizer, [state_text(row["verified_dead_state"])], 32)
    width = len(row["frontier"])
    legal = torch.tensor([CODEC.legal_mask(tuple(s)) for s in row["frontier"]], dtype=torch.bool)
    ref_targets = torch.zeros(width, len(CODEC.actions), dtype=torch.float32)
    successor_records = []
    seen_successors = set()
    for index, (actions, samples) in enumerate(zip(row["refutation_actions"], row["sampled_successors"])):
        ref_targets[index, actions] = 1.0
        for action_id, successor in samples:
            identity = tuple(successor)
            if identity not in seen_successors:
                seen_successors.add(identity)
                successor_records.append((index, action_id, state_text(successor)))
    succ_ids, succ_masks = tokenize(tokenizer, [record[2] for record in successor_records], 32)
    return {
        "row": row,
        "task_ids": task_ids,
        "task_mask": task_mask,
        "state_ids": state_ids.unsqueeze(0),
        "state_token_mask": state_tokens.unsqueeze(0),
        "state_mask": torch.ones(1, width, dtype=torch.bool),
        "refutation_ids": ref_ids.unsqueeze(0),
        "refutation_token_mask": ref_tokens.unsqueeze(0),
        "refutation_mask": torch.ones(1, 1, dtype=torch.bool),
        "action_mask": legal.unsqueeze(0),
        "survival_targets": torch.tensor(row["survival_targets"], dtype=torch.float32).unsqueeze(0),
        "action_targets": torch.tensor(row["action_targets"], dtype=torch.long).unsqueeze(0),
        "refutation_targets": ref_targets.unsqueeze(0),
        "successor_records": successor_records,
        "successor_ids": succ_ids.unsqueeze(0),
        "successor_mask": succ_masks.unsqueeze(0),
    }


def to_device(packed, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in packed.items()}


def forward_row(model, packed, coupling, training):
    data = to_device(packed, "cuda")
    output = model(
        task_ids=data["task_ids"], task_mask=data["task_mask"],
        state_ids=data["state_ids"], state_token_mask=data["state_token_mask"],
        state_mask=data["state_mask"], refutation_ids=data["refutation_ids"],
        refutation_token_mask=data["refutation_token_mask"],
        refutation_mask=data["refutation_mask"], action_mask=data["action_mask"],
        survival_targets=data["survival_targets"] if training else None,
        action_targets=data["action_targets"] if training else None,
        coupling=coupling,
    )
    if not training:
        return output.frontier, None
    legal = data["action_mask"]
    raw_match = output.frontier.action_refutation_logits.squeeze(-1)[legal]
    targets = data["refutation_targets"][legal]
    positive = targets.sum().clamp_min(1)
    negative = (targets.numel() - positive).clamp_min(1)
    match_loss = F.binary_cross_entropy_with_logits(
        raw_match.float(), targets.float(), pos_weight=(negative / positive).detach()
    )

    with torch.no_grad():
        teacher = model.state_encoder(data["successor_ids"], data["successor_mask"])[0].detach().float()
    records = data["successor_records"]
    pred = torch.stack([output.frontier.predicted_successors[0, index, action_id].float()
                        for index, action_id, _ in records])
    if len(records) > 1:
        pred = F.normalize(pred, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        contrastive = F.cross_entropy((pred @ teacher.T) / 0.1,
                                      torch.arange(len(records), device="cuda"))
    else:
        contrastive = F.smooth_l1_loss(pred, teacher)
    loss = output.loss + 0.5 * match_loss + 0.2 * contrastive
    return output.frontier, loss


@torch.no_grad()
def evaluate(model, packed_rows, coupling, limit):
    model.eval()
    chosen = packed_rows[:limit]
    successes, alive_total, refuted_chosen, refuted_opportunities = 0, 0, 0, 0
    selectivity, score_changes = [], []
    for packed in chosen:
        frontier, _ = forward_row(model, packed, coupling, training=False)
        data = to_device(packed, "cuda")
        blank = model(
            task_ids=data["task_ids"], task_mask=data["task_mask"],
            state_ids=data["state_ids"], state_token_mask=data["state_token_mask"],
            state_mask=data["state_mask"], refutation_ids=data["refutation_ids"],
            refutation_token_mask=data["refutation_token_mask"],
            refutation_mask=torch.zeros_like(data["refutation_mask"]),
            action_mask=data["action_mask"], coupling=coupling,
        ).frontier
        target = packed["row"]["target"]
        can_solve = solver(target)
        for index, state_values in enumerate(packed["row"]["frontier"]):
            action_id = int(frontier.proposal_logits[0, index].argmax())
            if packed["row"]["survival_targets"][index]:
                alive_total += 1
                successes += int(can_solve(CODEC.apply(tuple(state_values), action_id)))
            refuted = set(packed["row"]["refutation_actions"][index])
            if refuted:
                refuted_opportunities += 1
                refuted_chosen += int(action_id in refuted)
        legal = data["action_mask"]
        positive = data["refutation_targets"] > 0
        negative = legal & ~positive
        if (positive & legal).any() and negative.any():
            selectivity.append(float(frontier.action_refutation_match[positive & legal].float().mean()
                                     - frontier.action_refutation_match[negative].float().mean()))
            score_changes.append(float((frontier.proposal_logits - blank.proposal_logits)[positive & legal].float().mean()))
    return {
        "rows": len(chosen),
        "winning_action_rate_on_alive_states": successes / max(alive_total, 1),
        "alive_states": alive_total,
        "known_refuted_action_top1_rate": refuted_chosen / max(refuted_opportunities, 1),
        "refuted_action_opportunities": refuted_opportunities,
        "refutation_match_selectivity": statistics.mean(selectivity) if selectivity else None,
        "mean_score_change_on_refuted_actions": statistics.mean(score_changes) if score_changes else None,
    }


def train_arm(mode, train, validation, test, steps, eval_rows, output_dir, seed):
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = NativeFrontierLM(base, actions=len(CODEC.actions)).to(device="cuda", dtype=torch.bfloat16)
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (backbone_params if name.startswith("backbone.") else head_params).append(parameter)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 3e-4},
    ])
    backbone_parameter = model.backbone.model.layers[0].self_attn.q_proj.weight
    frontier_parameter = model.frontier_cell.transition_head[0].weight
    backbone_before = backbone_parameter.detach().clone()
    frontier_before = frontier_parameter.detach().clone()
    before = {
        "validation": evaluate(model, validation, mode, eval_rows),
        "test_5_numbers": evaluate(model, test, mode, eval_rows),
    }
    order = random.Random(71).choices(range(len(train)), k=steps)
    losses, step_times = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(order, 1):
        model.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        _, loss = forward_row(model, train[index], mode, training=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps({"mode": mode, "step": step, "loss": float(loss.detach()),
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
        losses.append(float(loss.detach()))
        step_times.append(time.perf_counter() - tick)
    model.eval()
    backbone_changed = not torch.equal(backbone_before, backbone_parameter.detach())
    frontier_changed = not torch.equal(frontier_before, frontier_parameter.detach())
    if not backbone_changed or not frontier_changed:
        raise RuntimeError("backbone or frontier weights did not change")
    after = {
        "validation": evaluate(model, validation, mode, eval_rows),
        "test_5_numbers": evaluate(model, test, mode, eval_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{mode}.pt"
    torch.save(model.state_dict(), checkpoint)
    with torch.no_grad():
        reference = forward_row(model, validation[0], mode, training=False)[0].proposal_logits.cpu()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        reloaded = forward_row(model, validation[0], mode, training=False)[0].proposal_logits.cpu()
    torch.testing.assert_close(reference, reloaded)
    result = {
        "mode": mode, "steps": steps, "seed": seed, "model": MODEL, "revision": REVISION,
        "training_status": "load-forward-backward-update-changed-weights-save-reload; behavior measured below",
        "before": before, "after": after,
        "train_loss_first": losses[0], "train_loss_last": losses[-1],
        "steady_step_seconds": statistics.median(step_times[1:]) if len(step_times) > 1 else step_times[0],
        "train_elapsed_s": time.perf_counter() - started,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "save_reload_equal": True,
        "backbone_changed": backbone_changed,
        "frontier_changed": frontier_changed,
    }
    del model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/native_countdown"))
    parser.add_argument("--out", type=Path, default=Path("runs/native_countdown_pilot"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-rows", type=int, default=50)
    parser.add_argument("--arms", default="zero,inhibit")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_rows < 1:
        parser.error("steps and eval rows must be positive")
    paths = {name: args.data / f"{name}.jsonl" for name in ("train", "validation", "test_5_numbers")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    packed = {name: [pack_row(row, tokenizer) for row in read_rows(path)]
              for name, path in paths.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "Countdown mechanism pilot; no source-disjoint or broad-reasoning claim",
        "data_sha256": {name: sha256(path) for name, path in paths.items()},
        "requested_arms": args.arms.split(","), "steps_per_arm": args.steps,
        "eval_rows_per_split": args.eval_rows,
        "arms": [],
    }
    for mode in args.arms.split(","):
        if mode not in ("zero", "inhibit"):
            parser.error("arms must be zero and/or inhibit")
        result = train_arm(mode, packed["train"], packed["validation"], packed["test_5_numbers"],
                           args.steps, args.eval_rows, args.out, args.seed)
        report["arms"].append(result)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed_arm": mode, "after": result["after"]}), flush=True)


if __name__ == "__main__":
    main()
