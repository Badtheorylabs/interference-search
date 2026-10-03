"""Measure cached native inference overhead against stock Qwen3-4B."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM


def load_native(args, tokenizer):
    base = AutoConfig.from_pretrained(args.model, local_files_only=True)
    values = base.to_dict(); values.pop("model_type", None); values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **values, frontier_slots=8, frontier_dim=128, frontier_heads=4,
        tool_error_token_id=tokenizer.convert_tokens_to_ids("<tool_response>"))
    model = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    model.model.reset_native_output(); model.model.set_native_dtype(torch.float32)
    if args.native_state:
        state = torch.load(args.native_state, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    return model.eval()


@torch.no_grad()
def one_run(model, ids, tokens, coupling=None):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    kwargs = {} if coupling is None else {"coupling": coupling}
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=True, logits_to_keep=1, **kwargs)
    torch.cuda.synchronize(); prefill = time.perf_counter() - started
    past = output.past_key_values
    next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    decode_started = time.perf_counter()
    for step in range(tokens):
        mask = torch.ones(ids.shape[0], ids.shape[1] + step + 1,
                          dtype=torch.long, device=ids.device)
        output = model(input_ids=next_id, attention_mask=mask,
                       past_key_values=past, use_cache=True,
                       logits_to_keep=1, **kwargs)
        past = output.past_key_values
        next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(); decode = time.perf_counter() - decode_started
    frontier_bytes = 0
    frontier = getattr(past, "frontier_state", None)
    if frontier is not None:
        frontier_bytes = frontier.numel() * frontier.element_size()
    return {"prefill_seconds": prefill, "decode_seconds": decode,
            "decode_tokens_per_second": tokens / decode,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
            "frontier_cache_bytes": frontier_bytes}


def summarize(rows):
    return {key: statistics.median(row[key] for row in rows)
            for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-state")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = ("Fix this Python function after its verified test failure:\n"
              "def total(values): return values[0]\n"
              "AssertionError: got 1, expected 6\n<tool_response>\n")
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    native = load_native(args, tokenizer)
    stock = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    # Untimed warmup.
    one_run(stock, ids, 4)
    one_run(native, ids, 4, "zero")
    one_run(native, ids, 4, "inhibit")
    raw = {"stock": [], "native_zero": [], "native_inhibit": []}
    for _ in range(args.repeats):
        raw["stock"].append(one_run(stock, ids, args.tokens))
        raw["native_zero"].append(one_run(native, ids, args.tokens, "zero"))
        raw["native_inhibit"].append(one_run(native, ids, args.tokens, "inhibit"))
    groups = {arm: summarize(rows) for arm, rows in raw.items()}
    stock_speed = groups["stock"]["decode_tokens_per_second"]
    for arm in ("native_zero", "native_inhibit"):
        groups[arm]["decode_speed_vs_stock"] = (
            groups[arm]["decode_tokens_per_second"] / stock_speed)
    report = {"status": "native cached decode benchmark",
              "tokens": args.tokens, "repeats": args.repeats,
              "prompt_tokens": ids.shape[1], "groups": groups, "raw": raw}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
