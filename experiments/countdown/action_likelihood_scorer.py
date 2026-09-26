"""Score legal structured actions with a stock model, then merge equal successors.

The environment provides legal arithmetic syntax, as in the released paper.
The model ranks actions in one batched scoring pass; actions that reach the
same state pool probability mass. Exact reachability is used only in the
evaluation below, never in the ranking.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.countdown import solver
from interference_search.countdown_actions import CountdownActionCodec
from interference_search.parallel_reasoning import logadd


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def prompt_for(problem, state, tokenizer):
    shown = ", ".join(f"{index}={value}" for index, value in enumerate(state))
    message = (
        f"Target: {problem['target']}. Current available numbers by zero-based position: {shown}. "
        "Choose exactly one move using two different positions and +, -, *, or /. "
        "The result must be a positive whole number. More moves may follow. "
        'Reply only as JSON: {"i":0,"op":"+","j":1}.'
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )


class ActionLikelihoodScorer:
    def __init__(self, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        ).eval()
        self.codec = CountdownActionCodec(max_numbers=7)
        self.batch_size = batch_size
        self.forward_calls = 0
        self.scored_actions = 0
        self.input_tokens = 0

    def score(self, problem, state):
        prompt = prompt_for(problem, state, self.tokenizer)
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        candidates = []
        for action_id, valid in enumerate(self.codec.legal_mask(state)):
            if not valid:
                continue
            action = self.codec.actions[action_id]
            text = json.dumps({"i": action.left, "op": action.op, "j": action.right}, separators=(",", ":"))
            ids = self.tokenizer.encode(prompt + text, add_special_tokens=False)
            if ids[:len(prefix)] != prefix:
                raise RuntimeError("candidate tokenization changed the prompt prefix")
            candidates.append((action_id, ids, len(prefix), self.codec.apply(state, action_id)))
        ranked = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            max_length = max(len(item[1]) for item in batch)
            token_ids = torch.full((len(batch), max_length), self.tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(token_ids)
            for index, (_, ids, _, _) in enumerate(batch):
                token_ids[index, :len(ids)] = torch.tensor(ids)
                attention[index, :len(ids)] = 1
            self.input_tokens += int(attention.sum())
            with torch.inference_mode():
                output = self.model(input_ids=token_ids.to("cuda"), attention_mask=attention.to("cuda"))
            self.forward_calls += 1
            for index, (action_id, ids, prefix_len, child) in enumerate(batch):
                positions = torch.arange(prefix_len - 1, len(ids) - 1, device="cuda")
                labels = torch.tensor(ids[prefix_len:], device="cuda")
                step_logits = output.logits[index, positions].float()
                selected = step_logits.log_softmax(dim=-1).gather(1, labels[:, None]).sum().item()
                ranked.append((action_id, child, selected, len(labels)))
        self.scored_actions += len(ranked)
        return ranked


def merged_successors(ranked):
    groups = {}
    for action_id, child, log_probability, token_count in ranked:
        old = groups.get(child)
        if old is None:
            groups[child] = {"log_mass": log_probability, "best_action": action_id,
                             "best_log_probability": log_probability, "paths": 1}
        else:
            old["log_mass"] = logadd(old["log_mass"], log_probability)
            old["paths"] += 1
            if log_probability > old["best_log_probability"]:
                old["best_action"] = action_id
                old["best_log_probability"] = log_probability
    return sorted(groups.items(), key=lambda item: (-item[1]["log_mass"], item[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/native_countdown/validation.jsonl"))
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-dead", action="store_true")
    args = parser.parse_args()
    if min(args.states, args.top_k, args.batch_size) < 1:
        parser.error("counts must be positive")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    scorer = ActionLikelihoodScorer(batch_size=args.batch_size)
    checked = []
    started = time.perf_counter()
    for row in rows:
        for state_values, alive in zip(row["frontier"], row["survival_targets"]):
            if not alive and not args.include_dead:
                continue
            state = tuple(state_values)
            ranked = scorer.score(row, state)
            groups = merged_successors(ranked)
            reachable = solver(row["target"])
            legal_children = list({child for _, child, _, _ in ranked})
            top_k = [child for child, _ in groups[:args.top_k]]
            max_ranked = sorted(groups, key=lambda item: (-item[1]["best_log_probability"], item[0]))
            top_k_max = [child for child, _ in max_ranked[:args.top_k]]
            random_width = min(args.top_k, len(legal_children))
            dead_count = sum(not reachable(child) for child in legal_children)
            random_success = 1 - (math.comb(dead_count, random_width) / math.comb(len(legal_children), random_width)
                                  if dead_count >= random_width else 0)
            checked.append({
                "target": row["target"], "state": state,
                "alive": bool(alive),
                "legal_actions": len(ranked), "unique_children": len(groups),
                "top1_live": reachable(top_k[0]),
                "topk_any_live": any(reachable(child) for child in top_k),
                "max_action_top1_live": reachable(top_k_max[0]),
                "max_action_topk_live": any(reachable(child) for child in top_k_max),
                "randomk_expected_live": random_success,
                "top_children": top_k,
            })
            if len(checked) == args.states:
                break
        if len(checked) == args.states:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    alive_rows = [row for row in checked if row["alive"]]
    print(json.dumps({
        "status": "model-ranked legal action preflight; exact solver used only for scoring",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "states": len(checked), "top_k": args.top_k,
        "alive_states": len(alive_rows),
        "pooled_mass_top1_live_on_alive": sum(row["top1_live"] for row in alive_rows),
        "pooled_mass_topk_live_on_alive": sum(row["topk_any_live"] for row in alive_rows),
        "max_action_top1_live_on_alive": sum(row["max_action_top1_live"] for row in alive_rows),
        "max_action_topk_live_on_alive": sum(row["max_action_topk_live"] for row in alive_rows),
        "randomk_expected_live_on_alive": sum(row["randomk_expected_live"] for row in alive_rows),
        "model_forward_calls": scorer.forward_calls,
        "scored_actions": scorer.scored_actions,
        "input_tokens_processed": scorer.input_tokens,
        "wall_seconds": round(elapsed, 3),
        "rows": checked,
    }, indent=2))


if __name__ == "__main__":
    main()
