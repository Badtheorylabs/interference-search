"""Measure full-backbone state encoding with and without a shared prompt KV.

This uses a fixed set of canonical Countdown states and the same model weights.
It isolates encoder cost from search-policy changes and checks vector parity.
"""

import argparse
import asyncio
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interference_search.countdown_actions import CountdownActionCodec
from interference_search.native_async_countdown import NativeAsyncCountdownDomain
from interference_search.native_lm import NativeFrontierLM


def collect_states(count):
    start = (3, 19, 24, 98)
    codec = CountdownActionCodec(4)
    states = list(dict.fromkeys(
        child for index, legal in enumerate(codec.legal_mask(start)) if legal
        for child in [codec.apply(start, index)]
    ))
    return states[:count]


async def measure(model, tokenizer, repetitions, prefix_repeat, state_count):
    base = "Numbers: [3, 19, 24, 98]. Target: 361. Search exact arithmetic paths. "
    problem = {"numbers": [24, 98, 19, 3], "target": 361,
               "task_text": base * prefix_repeat}
    states = collect_states(state_count)
    domains = {
        mode: NativeAsyncCountdownDomain(model, tokenizer, max_numbers=4,
                                         state_mode=mode)
        for mode in ("backbone_full", "backbone_prefix")
    }
    contexts = {mode: await domain.prepare(problem) for mode, domain in domains.items()}
    saved_prefix_length = contexts["backbone_prefix"].prompt_cache.get_seq_length()
    for mode, domain in domains.items():
        domain._encode_new_states(states, contexts[mode])
    full = contexts["backbone_full"].state_vectors
    cached = contexts["backbone_prefix"].state_vectors
    prompt_cache_unchanged = (
        contexts["backbone_prefix"].prompt_cache.get_seq_length() == saved_prefix_length)
    max_abs = max(float((full[state] - cached[state]).abs().max()) for state in states)
    min_cosine = min(float(torch.nn.functional.cosine_similarity(
        full[state].float(), cached[state].float(), dim=0)) for state in states)
    full_actions = domains["backbone_full"]._propose_sync(states, contexts["backbone_full"])
    prefix_actions = domains["backbone_prefix"]._propose_sync(states, contexts["backbone_prefix"])
    exact_action_lists = sum(
        [item.action for item in left] == [item.action for item in right]
        for left, right in zip(full_actions, prefix_actions)
    )
    top_one_matches = sum(left and right and left[0].action == right[0].action
                          for left, right in zip(full_actions, prefix_actions))
    timings = {mode: [] for mode in domains}
    for iteration in range(repetitions):
        order = ("backbone_full", "backbone_prefix") if iteration % 2 == 0 else (
            "backbone_prefix", "backbone_full")
        for mode in order:
            domain, context = domains[mode], contexts[mode]
            context.state_vectors.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            domain._encode_new_states(states, context)
            torch.cuda.synchronize()
            timings[mode].append((time.perf_counter() - started) * 1000)
    return {"states": len(states), "repetitions": repetitions,
            "prefix_tokens": len(contexts["backbone_full"].prefix_token_ids),
            "prompt_cache_unchanged": prompt_cache_unchanged,
            "max_abs_vector_delta": round(max_abs, 6),
            "min_cosine": round(min_cosine, 8),
            "exact_action_lists": exact_action_lists,
            "top_one_matches": top_one_matches,
            "median_ms": {mode: round(statistics.median(times), 3)
                          for mode, times in timings.items()},
            "timings_ms": {mode: [round(value, 3) for value in times]
                           for mode, times in timings.items()},
            "task_backbone_forwards": {mode: domain.task_backbone_forwards
                                       for mode, domain in domains.items()},
            "state_backbone_forwards": {mode: domain.state_backbone_forwards
                                        for mode, domain in domains.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--prefix-repeat", type=int, default=1)
    parser.add_argument("--state-count", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    torch.manual_seed(11)
    model = NativeFrontierLM(backbone, actions=48, heads=4).to(
        device="cuda", dtype=torch.bfloat16).eval()
    result = asyncio.run(measure(model, tokenizer, args.repetitions,
                                 args.prefix_repeat, args.state_count))
    result.update({"model": args.model, "random_head_seed": 11,
                   "device": torch.cuda.get_device_name(0),
                   "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2)})
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
