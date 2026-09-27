"""Online ablation of trained Qwen4B inside synchronized Interference Search."""

import argparse
import asyncio
import importlib
import json
import statistics
import time
import types
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from interference_search.native_interference_search import synchronized_search
from interference_search.native_kv_countdown import NativeKVCountdownDomain
from interference_search.native_kv_frontier import NativeKVFrontier


def restore_reference_qwen3(backbone):
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    qwen3 = importlib.reload(qwen3)
    backbone.model.rotary_emb = qwen3.Qwen3RotaryEmbedding(
        config=backbone.config).to(device="cuda")
    backbone.forward = types.MethodType(qwen3.Qwen3ForCausalLM.forward, backbone)
    backbone.model.forward = types.MethodType(qwen3.Qwen3Model.forward, backbone.model)
    for layer in backbone.model.layers:
        layer.forward = types.MethodType(qwen3.Qwen3DecoderLayer.forward, layer)
        layer.self_attn.forward = types.MethodType(qwen3.Qwen3Attention.forward, layer.self_attn)


def read_problems(path, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [{"numbers": row["numbers"], "target": row["target"]}
            for row in rows[:count]]


async def evaluate(model, tokenizer, actions, problems, mode, width, budget, fanout):
    coupling, merge = mode
    rows = []
    for index, problem in enumerate(problems):
        domain = NativeKVCountdownDomain(
            model, tokenizer, actions, max_numbers=7,
            actions_per_state=fanout, coupling=coupling)
        result = await synchronized_search(
            domain, problem, expansion_budget=budget, width=width,
            execution_workers=width * fanout, merge=merge)
        rows.append({"index": index, "problem": problem, "solved": result.solved,
                     "rounds": result.rounds, "expanded": result.expanded,
                     "executions": result.executions,
                     "merged_away": result.merged_away,
                     "wall_seconds": result.wall_seconds,
                     "trace": result.trace})
        print(json.dumps({"mode": f"{coupling}_merge_{merge}", "index": index,
                          "solved": result.solved, "rounds": result.rounds}), flush=True)
    return {
        "coupling": coupling, "merge": merge, "problems": len(rows),
        "solved": sum(row["solved"] for row in rows),
        "solve_rate": sum(row["solved"] for row in rows) / len(rows),
        "mean_rounds": statistics.mean(row["rounds"] for row in rows),
        "mean_expanded": statistics.mean(row["expanded"] for row in rows),
        "mean_executions": statistics.mean(row["executions"] for row in rows),
        "mean_merged_away": statistics.mean(row["merged_away"] for row in rows),
        "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--data", default="data/native_countdown/test_5_numbers.jsonl")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    peft, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=256,
        dtype=torch.bfloat16, load_in_4bit=False)
    backbone = peft.get_base_model()
    restore_reference_qwen3(backbone)
    model = NativeKVFrontier(backbone, 168, match_space="model").cuda().eval()
    model.cell.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    actions = torch.load(args.actions, map_location="cuda", weights_only=True)
    problems = read_problems(args.data, args.count)
    started = time.perf_counter()
    results = {}
    for mode in (("zero", True), ("inhibit", True), ("inhibit", False)):
        key = f"{mode[0]}_merge_{mode[1]}"
        results[key] = asyncio.run(evaluate(
            model, tokenizer, actions, problems, mode,
            args.width, args.budget, args.fanout))
    report = {"status": "online synchronized native Interference Search ablation",
              "count": len(problems), "width": args.width, "budget": args.budget,
              "fanout": args.fanout, "elapsed_seconds": time.perf_counter() - started,
              "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
