"""Standard autoregressive Qwen4B control for the native online gate."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from interference_search.countdown import check_answer, prompt_text


def read_problems(path, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [{"numbers": row["numbers"], "target": row["target"]}
            for row in rows[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", default="data/native_countdown/test_5_numbers.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024,
        dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    problems = read_problems(args.data, args.count)

    # Keep model load and kernel warmup outside the measured loop.
    warm = tokenizer("Continue: 1 + 1 =", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**warm, max_new_tokens=4, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)

    rows = []
    started = time.perf_counter()
    for start in range(0, len(problems), args.batch_size):
        batch_problems = problems[start:start + args.batch_size]
        prompts = []
        for problem in batch_problems:
            user = prompt_text(problem)
            if getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}], tokenize=False,
                    add_generation_prompt=True)
            else:
                prompt = user + "\n"
            prompts.append(prompt)
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        torch.manual_seed(args.seed + start)
        tick = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.95, top_k=20,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - tick
        for offset, problem in enumerate(batch_problems):
            index = start + offset
            continuation = output[offset, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            generated = int((continuation != tokenizer.pad_token_id).sum())
            solved = check_answer(text, problem["numbers"], problem["target"])
            row = {"index": index, "problem": problem, "solved": solved,
                   "generated_tokens": generated,
                   "amortized_wall_seconds": batch_seconds / len(batch_problems),
                   "batch_wall_seconds": batch_seconds, "text": text}
            rows.append(row)
            print(json.dumps({"label": args.label, "index": index, "solved": solved,
                              "tokens": row["generated_tokens"],
                              "amortized_seconds": round(row["amortized_wall_seconds"], 3)}), flush=True)

    elapsed = time.perf_counter() - started
    report = {
        "status": "standard autoregressive Countdown control",
        "label": args.label, "model": args.model, "problems": len(rows),
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "seed": args.seed, "batch_size": args.batch_size,
        "solved": sum(row["solved"] for row in rows),
        "solve_rate": sum(row["solved"] for row in rows) / len(rows),
        "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in rows),
        "mean_amortized_wall_seconds": statistics.mean(
            row["amortized_wall_seconds"] for row in rows),
        "elapsed_seconds": elapsed, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
