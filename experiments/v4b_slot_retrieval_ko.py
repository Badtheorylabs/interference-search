"""V4b causal intervention at the supervised read path.

The null-split diagnostic showed near-zero full-vocabulary effect of slot
knockout on next-token logits at decision points (KL ~ 1e-9). The V4a-families
trainer never included an LM objective for the token readout path
(read_gate/token_up): its losses supervised the slot read heads
(decoded_hypotheses, slot_alive, slot_values) and family/equivalence
structure. So a generation-side null is predicted by the training design and
says nothing about slot causal control.

This diagnostic measures the causal effect at the path that WAS supervised:
slot knockout -> retrieval of the assigned candidate via decoded_hypotheses.

Interventions:
  normal:     retrieve all candidates, count correct (Hungarian assignment).
  KO assigned slot of candidate c: candidate c should fail retrieval while
  unrelated candidates remain correct.
  KO unrelated slot: the assigned candidate for another slot should be
  unaffected unless slots share it.

Record: retrieval vector (4-way) under each intervention, and the full
assignment cosine matrix. Selective effect = how much the KO'd candidate's
retrieval drops versus others.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interference_search.modeling_interference_qwen3 import (
    InterferenceQwen3ForCausalLM, NativeHypothesisUpdate)
from interference_search.configuration_interference_qwen3 import (
    InterferenceQwen3Config)
from experiments.midtrain_v4a_families import (
    assignment, select_candidates, task_tokens, teacher_targets)

_ORIG = NativeHypothesisUpdate.forward


def patched(self, hidden_states, frontier_state, refutation_mask=None,
            coupling="inhibit"):
    action = getattr(self, "_v4b_action", None)
    if action is not None:
        frontier_state = action(frontier_state)
    return _ORIG(self, hidden_states, frontier_state, refutation_mask, coupling)


NativeHypothesisUpdate.forward = patched


def clear(model):
    for layer in model.model.layers:
        update = getattr(layer, "hypothesis_update", None)
        if update is None:
            continue
        update._v4b_action = None


def set_action(model, action):
    for layer in model.model.layers:
        update = getattr(layer, "hypothesis_update", None)
        if update is not None:
            update._v4b_action = action


def zero_slot_action(slot):
    def action(frontier_state):
        revised = frontier_state.clone()
        revised[:, slot, :] = 0.0
        return revised
    return action


def sham_slot_action(slot, generator):
    """Norm-matched semantic destruction: replace slot content with a random
    unit vector scaled to the original norm. This controls for the
    "removed a large vector" / zero-norm degenerate-cosine explanation: the
    slot keeps its magnitude but loses the specific hypothesis content."""
    def action(frontier_state):
        revised = frontier_state.clone()
        cur = revised[:, slot, :]
        cur_n = cur.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rand = torch.randn(cur.shape, device=cur.device, dtype=cur.dtype)
        rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        revised[:, slot, :] = rand * cur_n
        return revised
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--revision", default="906bfd4b4dc7f14ee4320094d8b41684abff8539")
    ap.add_argument("--native-state", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--max-tasks", type=int, default=62)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision,
                                              local_files_only=True)
    base = AutoConfig.from_pretrained(args.base_model, revision=args.revision,
                                      local_files_only=True)
    values = base.to_dict(); values.pop("model_type", None); values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_call>"))
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.base_model, config=config, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.model.reset_native_output(); model.model.set_native_dtype(torch.float32)
    state = torch.load(Path(args.native_state), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines()]
    rng = random.Random(1234)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []

    with torch.no_grad():
        for row in rows[:args.max_tasks]:
            selected = select_candidates(row, rng)
            targets, alive, vals = teacher_targets(teacher, tokenizer, row, selected, device)
            ids, mask = task_tokens(tokenizer, row, device)

            clear(model)
            output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           coupling="zero")
            predicted = output.decoded_hypotheses[0].float()
            slots, sim = assignment(predicted, targets)
            retrieval = sim[slots].argmax(dim=-1)
            normal_correct = (retrieval == torch.arange(len(selected), device=device))

            # Knock out each candidate's assigned slot in turn; measure how
            # retrieval changes for all candidates.
            per_slot_ko = []
            per_slot_sham = []
            for c in range(len(selected)):
                ko_slot = int(slots[c].item())
                donor_slot = int(slots[(c + 1) % len(selected)].item())
                set_action(model, zero_slot_action(ko_slot))
                out_ko = model(input_ids=ids, attention_mask=mask, use_cache=False,
                               coupling="zero")
                pred_ko = out_ko.decoded_hypotheses[0].float()
                sim_ko = F.normalize(pred_ko.float(), dim=-1) @ F.normalize(
                    targets.float(), dim=-1).T
                retrieval_ko = sim_ko[slots].argmax(dim=-1)
                correct_ko = (retrieval_ko == torch.arange(len(selected), device=device))
                cos_before = float(sim[slots[c], c])
                cos_after = float(sim_ko[slots[c], c])
                per_slot_ko.append({
                    "ko_candidate": c, "ko_slot": ko_slot,
                    "correct": correct_ko.tolist(),
                    "self_cosine_before": cos_before, "self_cosine_after": cos_after,
                })
                clear(model)
                # norm-matched sham: destroy content, keep energy
                set_action(model, sham_slot_action(ko_slot, None))
                out_sham = model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 coupling="zero")
                pred_sham = out_sham.decoded_hypotheses[0].float()
                sim_sham = F.normalize(pred_sham.float(), dim=-1) @ F.normalize(
                    targets.float(), dim=-1).T
                retrieval_sham = sim_sham[slots].argmax(dim=-1)
                correct_sham = (retrieval_sham == torch.arange(len(selected), device=device))
                cos_sham = float(sim_sham[slots[c], c])
                per_slot_sham.append({
                    "sham_candidate": c, "ko_slot": ko_slot, "donor_slot": donor_slot,
                    "correct": correct_sham.tolist(),
                    "self_cosine_before": cos_before, "self_cosine_after_sham": cos_sham,
                })
                clear(model)

            results.append({
                "task_id": row["task_id"],
                "normal_correct": normal_correct.tolist(),
                "slot_assignments": [int(s) for s in slots.tolist()],
                "per_slot_ko": per_slot_ko,
                "per_slot_sham": per_slot_sham,
            })
            if len(results) % 10 == 0:
                print(f"done {len(results)}", flush=True)

    # Aggregate: for each task, KO of candidate c's slot should destroy c's
    # retrieval most, and destroy others less.
    assigned_self_drops, assigned_destroy, other_destroy_rate = [], [], []
    sham_self_drops, sham_destroy = [], []
    for r in results:
        n = len(r["normal_correct"])
        for entry in r["per_slot_ko"]:
            c = entry["ko_candidate"]
            drop_self = entry["self_cosine_before"] - entry["self_cosine_after"]
            destroyed_c = not entry["correct"][c]
            others_destroyed = sum(1 for i, ok in enumerate(entry["correct"])
                                   if i != c and not ok)
            assigned_self_drops.append(drop_self)
            assigned_destroy.append(destroyed_c)
            other_destroy_rate.append(others_destroyed / max(n - 1, 1))
        for entry in r["per_slot_sham"]:
            c = entry["sham_candidate"]
            drop_sham = entry["self_cosine_before"] - entry["self_cosine_after_sham"]
            sham_destroy.append(not entry["correct"][c])
            sham_self_drops.append(drop_sham)
    def rate(xs):
        return sum(1 for x in xs if x) / max(len(xs), 1) if xs else None
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / max(len(xs), 1) if xs else None
    summary = {
        "n_tasks": len(results),
        "ko_self_cosine_drop": mean(assigned_self_drops),
        "sham_self_cosine_drop": mean(sham_self_drops),
        "ko_assigned_destroy_rate": rate(assigned_destroy),
        "sham_assigned_destroy_rate": rate(sham_destroy),
        "other_destroy_rate": rate([x for x in other_destroy_rate if x is not None]),
    }
    (out / "report.json").write_text(json.dumps({"summary": summary, "per_task": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
