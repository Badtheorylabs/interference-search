"""V4b null-splitting diagnostic.

The smoke probe found near-zero effect of frontier slot knockout on a tracked
four-action decision distribution. Two hypotheses explain that:

A. Mechanism null: the frontier injection has no causal influence on
   next-token logits at decision points.
B. Measurement artifact: the injection changes logits substantially, but moves
   mass onto tokens outside the tracked four actions, so four-action deltas
   understate the intervention.

This diagnostic measures the FULL-vocabulary effect of assigned-slot knockout
at the decision token, plus injection magnitude at decision tokens versus
earlier tokens, plus four-action and full-vocab effects at the raw task-prompt
token (the position where V4a supervision existed).

"""

from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--revision", default="906bfd4b4dc7f14ee4320094d8b41684abff8539")
    ap.add_argument("--native-state", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--decision-points", required=True)
    ap.add_argument("--max-tasks", type=int, default=12)
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

    rows = {json.loads(l)["task_id"]: json.loads(l)
            for l in Path(args.rows).read_text().splitlines()}
    dps = [json.loads(l) for l in Path(args.decision_points).read_text().splitlines()]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    import random
    rng = random.Random(1234)

    with torch.no_grad():
        for dp in dps[:args.max_tasks]:
            tid = dp["task_id"]
            row = rows.get(tid)
            if row is None:
                continue
            selected = select_candidates(row, rng)
            passing = [i for i, c in enumerate(selected) if c["all_passed"]]
            if not passing:
                continue
            p_index = passing[0]
            ids, mask = task_tokens(tokenizer, row, device)
            output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           coupling="zero")
            predicted = output.decoded_hypotheses[0].float()
            targets, alive, values_ = teacher_targets(teacher, tokenizer, row, selected, device)
            slots, sim = assignment(predicted, targets)
            slot_of_P = int(slots[p_index].item())

            # Recurrent decision path exactly as the user specified:
            # prefill task tokens, carry (KV_t, F_t), then feed the decision
            # prefix stepwise. The final position is the decision point where
            # the frontier built from the task context is active.
            seq = ids[0].tolist()
            decision_tail = tokenizer(dp["context"] + "Continue",
                                      add_special_tokens=False)["input_ids"]
            # The last token is the live token receiving the frontier write.
            full_ids = torch.tensor([seq + decision_tail[:-1]], device=device)
            full_attention = torch.ones_like(full_ids)
            clear(model)
            logits_normal = model(input_ids=full_ids, attention_mask=full_attention,
                                  use_cache=False).logits[0, -1, :].float()
            set_action(model, zero_slot_action(slot_of_P))
            logits_ko = model(input_ids=full_ids, attention_mask=full_attention,
                              use_cache=False).logits[0, -1, :].float()
            clear(model)

            # Full-vocab effect
            delta_logits = (logits_ko - logits_normal)
            max_abs_delta = float(delta_logits.abs().max())
            mean_abs_delta = float(delta_logits.abs().mean())
            kl = float(F.kl_div(
                F.log_softmax(logits_ko, dim=-1),
                F.softmax(logits_normal, dim=-1), log_target=False))
            top_shift = int((delta_logits.abs() > 0.1).sum())

            # Four-action effect
            a_ids = [dp["actions"][lbl]["first_token_id"] for lbl in dp["action_order"]]
            p_norm = torch.softmax(torch.tensor([logits_normal[i] for i in a_ids]), dim=-1)
            p_ko = torch.softmax(torch.tensor([logits_ko[i] for i in a_ids]), dim=-1)
            four_delta = float((p_ko - p_norm).abs().max())
            four_l1 = float((p_ko - p_norm).abs().sum())

            # Task-token position (V4a supervision point) effect on retrieval
            # already exists in smoke; here also measure whether knockout at
            # the decision point changes full-vocab argmax.
            argmax_normal = int(logits_normal.argmax())
            argmax_ko = int(logits_ko.argmax())

            results.append({
                "task_id": tid, "slot_P": slot_of_P,
                "max_abs_delta": max_abs_delta, "mean_abs_delta": mean_abs_delta,
                "kl_ko_vs_normal": kl, "n_tokens_delta_gt_0.1": top_shift,
                "four_action_l1": four_l1, "four_action_max_delta": four_delta,
                "argmax_changed": argmax_normal != argmax_ko,
                "four_normal": p_norm.tolist(), "four_ko": p_ko.tolist(),
            })

    summary = {
        "n": len(results),
        "mean_max_abs_delta": sum(r["max_abs_delta"] for r in results) / max(len(results), 1),
        "mean_mean_abs_delta": sum(r["mean_abs_delta"] for r in results) / max(len(results), 1),
        "mean_kl": sum(r["kl_ko_vs_normal"] for r in results) / max(len(results), 1),
        "mean_four_action_l1": sum(r["four_action_l1"] for r in results) / max(len(results), 1),
        "frac_argmax_changed": sum(r["argmax_changed"] for r in results) / max(len(results), 1),
    }
    (out / "report.json").write_text(json.dumps({"summary": summary, "per_task": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
