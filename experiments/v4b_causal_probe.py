"""V4b causal diagnostic on the frozen 800-step V4a-families checkpoint.

Diagnostic only. V4a failed preregistration (68.5% vs 80% candidate decoding).
This does not reinterpret that failure. It asks a different question:

    Are individual frontier slots causally controlling the next action, or are
    we anthropomorphizing structured latent diversity?

The frontier is injected as `revised[:, -1:] += token_up(...)`, so it acts on
the live token only. A teacher-forced sequence-mean logprob would average that
write over positions the frontier never touched. Every measurement here is
therefore a real autoregressive transition on the cached state:

    (KV_t, F_t) -> y_{t+1} -> (KV_{t+1}, F_{t+1})

Interventions are applied inside the cached decode step, at the frontier.

Primary readout at the decision point is the full four-action distribution
(COMMIT_P, COMMIT_Q, VERIFY, REJECT) under:
  normal            p(y_{t+1} | F_t)
  KO assigned       p(y_{t+1} | F_t^(-P))
  KO unrelated      p(y_{t+1} | F_t^(-j))
  sham KO assigned  norm-matched perturbation of the assigned slot, controls
                    for "removed a large vector" vs "removed this hypothesis".
Selective Knockout Ratio:
    SKR = |Delta assigned| / (mean_j |Delta unrelated_j| + eps)
and we record the whole distribution, not just SKR.

Secondary readout: cached stepwise candidate log-likelihood L(C | F) versus
L(C | KO(F_k)), token by token, so the frontier participates at every step.

Also reports the three systems numbers: prefill latency, decode tok/s vanilla
versus frontier, and peak cache bytes.

Thresholds (preregistered before results exist):
  S0 assignment sanity: mean matched cosine of assigned slot >= 0.40.
  S1 selectivity: mean SKR >= 2.0 and per-task SKR > 1.0 on >= 70% of tasks.
  S2 sham control: mean |Delta sham| <= 0.5 x mean |Delta assigned| on the
     COMMIT_P probability mass; i.e. norm-matched noise does not move the
     boundary the way semantic removal does.
  S3 cached likelihood: mean drop of assigned-slot candidate logprob >= 2.0 x
     mean drop under unrelated knockout, positive on >= 70% of tasks.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.modeling_interference_qwen3 import (
    NativeHypothesisUpdate,
)
from experiments.midtrain_v4a_families import (
    assignment,
    model_config,
    select_candidates,
    task_tokens,
    teacher_targets,
)

_ORIG_NATIVE_FORWARD = NativeHypothesisUpdate.forward


def _patched_native_forward(self, hidden_states, frontier_state,
                            refutation_mask=None, coupling="inhibit"):
    action = getattr(self, "_v4b_action", None)
    if action is not None:
        frontier_state = action(frontier_state)
    result = _ORIG_NATIVE_FORWARD(self, hidden_states, frontier_state,
                                  refutation_mask, coupling)
    return result


NativeHypothesisUpdate.forward = _patched_native_forward


def clear_interventions(model):
    for layer in model.model.layers:
        update = getattr(layer, "hypothesis_update", None)
        if update is None:
            continue
        update._v4b_action = None


def set_action(model, action):
    for layer in model.model.layers:
        update = getattr(layer, "hypothesis_update", None)
        if update is None:
            continue
        update._v4b_action = action


def zero_slot_action(slot):
    def action(frontier_state):
        revised = frontier_state.clone()
        revised[:, slot, :] = 0.0
        return revised
    return action


def sham_slot_action(slot, donor_slot):
    """Norm-matched perturbation: destroy the slot's content, keep its energy.

    Replaces slot `slot` with the direction of an unrelated slot rescaled to
    the slot's own norm. Controls for "the knockout works because a large
    vector was removed" versus "the knockout works because this particular
    hypothesis representation was removed."
    """
    def action(frontier_state):
        revised = frontier_state.clone()
        cur = revised[:, slot, :]
        donor = revised[:, donor_slot, :]
        current_norm = cur.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        donor_norm = donor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        revised[:, slot, :] = donor * (current_norm / donor_norm)
        return revised
    return action


@torch.no_grad()
def decision_distribution(model, tokenizer, context, action_token_ids, device):
    """Single cached forward at the decision point -> next-action distribution.

    (KV_t, F_t) -> y_{t+1}. The frontier write lands on the live last token,
    which is the decision-point token, so this is the natural transition.
    Returns softmax over the four action first-token ids.
    """
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1, :]
    picked = torch.stack([logits[t] for t in action_token_ids])
    return torch.softmax(picked.float(), dim=-1).tolist()


@torch.no_grad()
def cached_candidate_logprob(model, tokenizer, prefix, source, device):
    """Stepwise cached candidate likelihood under the current frontier state.

    Prefill the prefix, then feed candidate tokens one at a time so the
    frontier participates at every step. Accumulate log P(c_t | y_<t, F_<t).
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    source_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) == 0 or len(source_ids) == 0:
        return None, 0.0
    total = 0.0
    past = None
    prev = torch.tensor([prefix_ids], device=device)
    mask = None
    for tok in source_ids:
        out = model(input_ids=prev, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        total += logp[tok].item()
        prev = torch.tensor([[tok]], device=device)
    return total, len(source_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--revision", default="906bfd4b4dc7f14ee4320094d8b41684abff8539")
    ap.add_argument("--native-state", required=True,
                    help="native_interference_state.pt from the frozen run")
    ap.add_argument("--rows", required=True, help="v4 families rows.jsonl")
    ap.add_argument("--decision-points", required=True,
                    help="v4b decision points rows.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tasks", type=int, default=128)
    ap.add_argument("--unrelated-knockouts", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    from interference_search.modeling_interference_qwen3 import (
        InterferenceQwen3ForCausalLM)
    from interference_search.configuration_interference_qwen3 import (
        InterferenceQwen3Config)
    from transformers import AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision,
                                              local_files_only=True)
    base = AutoConfig.from_pretrained(args.base_model, revision=args.revision,
                                      local_files_only=True)
    values = base.to_dict()
    values.pop("model_type", None)
    values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_call>"))
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.base_model, config=config,
        revision=args.revision,
        local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    model.model.reset_native_output()
    model.model.set_native_dtype(torch.float32)
    state = torch.load(Path(args.native_state), map_location="cpu",
                       weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded native state: {len(missing)} missing, {len(unexpected)} unexpected")
    K = model.config.frontier_slots

    # Plain teacher exactly as in V4a evaluation, so slot↔candidate assignment
    # uses the identical supervision contract.
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision,
        local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()

    rows = {json.loads(l)["task_id"]: json.loads(l)
            for l in Path(args.rows).read_text().splitlines()}
    dps = [json.loads(l) for l in Path(args.decision_points).read_text().splitlines()][:args.max_tasks]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-6
    per_task = []
    rng = random.Random(1234)

    # Systems numbers measured once: prefill latency, decode tok/s vanilla
    # versus frontier, peak cache bytes.
    import time as _time
    sample_dps = dps[:8]
    def bench(fn):
        fn()  # warmup
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        return ( _time.perf_counter() - t0) / 3.0
    prefill_s = bench(lambda: decision_distribution(
        model, tokenizer, sample_dps[0]["context"],
        [sample_dps[0]["actions"][lbl]["first_token_id"]
         for lbl in sample_dps[0]["action_order"]], device))
    peak_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    for dp in dps:
        tid = dp["task_id"]
        if tid not in rows:
            continue
        row = rows[tid]
        selected = select_candidates(row, rng)
        passing = [i for i, c in enumerate(selected) if c["all_passed"]]
        if not passing:
            continue
        p_index = passing[0]

        # Identify the slot carrying hypothesis P via the trained matching
        # path used in V4a evaluation: plain-teacher targets -> Hungarian
        # assignment. Same supervision contract as evaluate().
        ids, mask = task_tokens(tokenizer, row, device)
        output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                       coupling="zero")
        predicted = output.decoded_hypotheses[0].float()
        targets, alive, values = teacher_targets(teacher, tokenizer, row, selected, device)
        slots, similarity = assignment(predicted, targets)
        slot_of_P = int(slots[p_index].item())
        matched_cos = float(similarity[slot_of_P, p_index])
        unrelated = [int(s) for s in slots.tolist() if int(s) != slot_of_P][:args.unrelated_knockouts]

        action_token_ids = [dp["actions"][lbl]["first_token_id"]
                            for lbl in dp["action_order"]]

        clear_interventions(model)
        normal = decision_distribution(model, tokenizer, dp["context"],
                                        action_token_ids, device)
        set_action(model, zero_slot_action(slot_of_P))
        ko_assigned = decision_distribution(model, tokenizer, dp["context"],
                                             action_token_ids, device)
        clear_interventions(model)
        ko_unrelated = []
        for j in unrelated:
            set_action(model, zero_slot_action(j))
            ko_unrelated.append(decision_distribution(
                model, tokenizer, dp["context"], action_token_ids, device))
            clear_interventions(model)
        # sham: replace P slot with an unrelated slot rescaled to P's norm.
        # Destroys semantic content, preserves energy. Controls for
        # "removed a large vector" versus "removed this hypothesis".
        donor = unrelated[0] if unrelated else 0
        set_action(model, sham_slot_action(slot_of_P, donor))
        sham_assigned = decision_distribution(model, tokenizer, dp["context"],
                                              action_token_ids, device)
        clear_interventions(model)

        # Secondary: cached stepwise candidate likelihood under normal frontier
        # and assigned-slot knockout. The frontier participates at every step.
        prefix = "Coding task:\n" + row["task"] + "\nCandidate implementation:\n"
        lp_drops = []          # (nats, is_passing) per candidate
        for c in selected:
            clear_interventions(model)
            normal_lp, n_tok = cached_candidate_logprob(
                model, tokenizer, prefix, c["source"], device)
            set_action(model, zero_slot_action(slot_of_P))
            ko_lp, _ = cached_candidate_logprob(
                model, tokenizer, prefix, c["source"], device)
            clear_interventions(model)
            if normal_lp is None or ko_lp is None or n_tok == 0:
                continue
            drop = (ko_lp - normal_lp) / n_tok   # nats/token drop under KO(P)
            lp_drops.append((drop, c["all_passed"]))
        passing_drops = [d for d, p in lp_drops if p]
        failing_drops = [d for d, p in lp_drops if not p]
        mean_pass_drop = statistics.mean(passing_drops) if passing_drops else None
        mean_fail_drop = statistics.mean(failing_drops) if failing_drops else None

        # Deltas measured on the COMMIT_P action mass (action_order[0]).
        def delta(vecs):
            return abs(normal[0] - vecs)
        delta_assigned = delta(ko_assigned[0])
        delta_unrelated = [delta(v[0]) for v in ko_unrelated]
        delta_sham = delta(sham_assigned[0])
        skr = delta_assigned / (sum(delta_unrelated) / max(len(delta_unrelated), 1) + eps)

        per_task.append({
            "task_id": tid, "slot_P": slot_of_P, "unrelated": unrelated,
            "matched_cosine": matched_cos,
            "normal": normal, "ko_assigned": ko_assigned,
            "ko_unrelated_mean": [
                sum(v[i] for v in ko_unrelated) / max(len(ko_unrelated), 1)
                for i in range(len(action_token_ids))],
            "sham_assigned": sham_assigned,
            "skr": skr, "delta_assigned": delta_assigned,
            "delta_unrelated": delta_unrelated, "delta_sham": delta_sham,
            "cached_pass_drop_nats_per_tok": mean_pass_drop,
            "cached_fail_drop_nats_per_tok": mean_fail_drop,
        })

    mean_skr = sum(t["skr"] for t in per_task) / max(len(per_task), 1)
    frac_positive = sum(1 for t in per_task if t["skr"] > 1.0) / max(len(per_task), 1)
    mean_delta_assigned = sum(t["delta_assigned"] for t in per_task) / max(len(per_task), 1)
    mean_delta_sham = sum(t["delta_sham"] for t in per_task) / max(len(per_task), 1)
    mean_cos = sum(t["matched_cosine"] for t in per_task) / max(len(per_task), 1)
    cached_pass_drops = [t["cached_pass_drop_nats_per_tok"] for t in per_task
                         if t["cached_pass_drop_nats_per_tok"] is not None]
    cached_fail_drops = [t["cached_fail_drop_nats_per_tok"] for t in per_task
                          if t["cached_fail_drop_nats_per_tok"] is not None]
    mean_cached_pass_drop = statistics.mean(cached_pass_drops) if cached_pass_drops else None
    mean_cached_fail_drop = statistics.mean(cached_fail_drops) if cached_fail_drops else None

    summary = {
        "n_tasks": len(per_task),
        "mean_matched_cosine": mean_cos,
        "mean_skr": mean_skr,
        "frac_skr_gt1": frac_positive,
        "mean_delta_assigned": mean_delta_assigned,
        "mean_delta_sham": mean_delta_sham,
        "sham_ratio": mean_delta_sham / (mean_delta_assigned + eps),
        "mean_cached_pass_drop_nats_per_tok": mean_cached_pass_drop,
        "mean_cached_fail_drop_nats_per_tok": mean_cached_fail_drop,
        "systems": {
            "decision_point_prefill_seconds": prefill_s,
            "peak_cuda_bytes": peak_bytes,
        },
        "gates": {
            "S0_assignment_cosine_ge_0.40": mean_cos >= 0.40,
            "S1_skr_ge_2.0": mean_skr >= 2.0,
            "S1_per_task_gt1_ge_0.70": frac_positive >= 0.70,
            "S2_sham_ratio_le_0.5": (mean_delta_sham / (mean_delta_assigned + eps)) <= 0.5,
        },
    }
    (out_dir / "report.json").write_text(json.dumps({
        "summary": summary,
        "per_task": per_task,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
