"""Execute standard-generate code repairs from the native 4B checkpoint."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.mbpp_async_domain import extract_code
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM
from interference_search.sandboxed_python import BubblewrapPython


def prompt_for(row):
    failed = row["negatives"][0]
    feedback = "\n".join(
        f"{status}: {detail}" for status, detail in failed["tests"])
    return (
        f"Coding task:\n{row['task']}\n"
        f"Failed implementation:\n{failed['source']}\n"
        f"Verified execution failure:\n{feedback}\n"
        "<tool_response>\n"
        "Produce the corrected Python implementation. Return only Python source code.\n")


def load_native(args, tokenizer):
    base = AutoConfig.from_pretrained(args.model, local_files_only=True)
    values = base.to_dict()
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
    state = torch.load(args.native_state, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    native_names = set(state)
    if unexpected or any(name in native_names for name in missing):
        raise RuntimeError("native checkpoint did not reload cleanly")
    return model.eval()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, coupling=None):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    kwargs = {} if coupling is None else {"coupling": coupling}
    started = time.perf_counter()
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs)
    torch.cuda.synchronize()
    continuation = output[0, inputs["input_ids"].shape[1]:]
    return (tokenizer.decode(continuation, skip_special_tokens=True),
            int(continuation.numel()), time.perf_counter() - started)


async def evaluate(args):
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    split = max(1, int(len(rows) * 0.8))
    rows = rows[split:split + args.count]
    mbpp = {row["task_id"]: row for row in json.loads(
        Path("data/mbpp_sanitized.json").read_text())}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    native = load_native(args, tokenizer)
    stock = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    sandbox = BubblewrapPython()
    results = {arm: [] for arm in ("native_inhibit", "native_zero", "stock")}
    for index, row in enumerate(rows):
        prompt = prompt_for(row)
        for arm in results:
            if arm == "native_inhibit":
                text, tokens, seconds = generate(
                    native, tokenizer, prompt, args.max_new_tokens, "inhibit")
            elif arm == "native_zero":
                text, tokens, seconds = generate(
                    native, tokenizer, prompt, args.max_new_tokens, "zero")
            else:
                text, tokens, seconds = generate(
                    stock, tokenizer, prompt, args.max_new_tokens)
            source = extract_code(text)
            outcome = await sandbox.run(
                source, row["tests"], mbpp[row["task_id"]].get("test_imports", []))
            result = {"task_id": row["task_id"], "solved": outcome.all_passed,
                      "passed": outcome.passed, "tests": len(outcome.tests),
                      "tokens": tokens, "seconds": seconds,
                      "source": source, "outcome": [list(test) for test in outcome.tests]}
            results[arm].append(result)
            print(json.dumps({"index": index, "task_id": row["task_id"],
                              "arm": arm, "solved": outcome.all_passed,
                              "passed": outcome.passed}), flush=True)
    groups = {}
    for arm, arm_rows in results.items():
        groups[arm] = {
            "solved": sum(row["solved"] for row in arm_rows),
            "total": len(arm_rows),
            "mean_tests_passed": statistics.mean(row["passed"] for row in arm_rows),
            "mean_tokens": statistics.mean(row["tokens"] for row in arm_rows),
            "mean_seconds": statistics.mean(row["seconds"] for row in arm_rows),
        }
    return {"status": "standard-generate verified repair gate",
            "native_state": args.native_state, "groups": groups, "rows": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-state", required=True)
    parser.add_argument("--data", default="data/mbpp_same_state_118/rows.jsonl")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
