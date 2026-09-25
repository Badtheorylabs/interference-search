"""Chain of thought timed one problem at a time, to compare against probe.py's search on equal terms.

Same model, prompt, sampler and 1,500-token budget as the transcript arm of llm_state.py, on the same 30
hard problems (seed 11), but each problem runs alone, the way probe.py times its search.
"""
import json
import random
import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from interference_search.countdown import check_answer, gen_hard_problem, prompt_text

model, tok = load("mlx-community/Qwen3-1.7B-4bit")
sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20)
rng = random.Random(11)
problems = [gen_hard_problem(rng) for _ in range(30)]

generate(model, tok, prompt="hi", max_tokens=8)            # warm up kernels outside the timed loop
rows = []
for i, p in enumerate(problems):
    pr = tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}], add_generation_prompt=True, tokenize=True)
    t = time.perf_counter()
    text = generate(model, tok, prompt=pr, max_tokens=1500, sampler=sampler)
    sec = time.perf_counter() - t
    rows.append({"solved": check_answer(text, p["numbers"], p["target"]), "sec": sec, "gen": len(tok.encode(text))})
    print(f"{i + 1:2d}/30  {sec:5.1f} s  {rows[-1]['gen']:5d} tokens  solved {rows[-1]['solved']}", flush=True)

solved = sum(r["solved"] for r in rows)
mean = sum(r["sec"] for r in rows) / len(rows)
print(f"chain of thought, one problem at a time: solved {solved}/30, {mean:.1f} s/problem")
json.dump({"solved": solved, "seconds_per_problem": mean, "per_problem": rows}, open("cot_timing.json", "w"), indent=1)
