"""Real chain-of-thought from Qwen3-1.7B on the 4 showcase problems, with every attempt it writes.

Each attempt is a full expression that uses all four numbers, pulled from the reasoning with the same
exact extractor used in Phase 0. Recorded: token position, expression, value, correct, and whether the
model had already tried the same expression earlier in its own reasoning."""
import json
import bisect
from pathlib import Path
import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from interference_search.countdown import check_answer, extract_attempts, prompt_text

show = json.load(open(Path(__file__).resolve().parents[2] / "results" / "llm" / "viz_slim.json"))["showcase"]
model, tok = load("mlx-community/Qwen3-1.7B-4bit")
mx.random.seed(7)
prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_text({"numbers": p["numbers"], "target": p["target"]})}],
                                   add_generation_prompt=True, tokenize=True) for p in show]
r = batch_generate(model, tok, prompts, max_tokens=2048, sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20), verbose=False)
out = []
for p, text in zip(show, r.texts):
    enc = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = [a for a, _ in enc["offset_mapping"]]
    seen, atts = set(), []
    for a in extract_attempts(text, p["numbers"], p["target"]):
        atts.append({"tok": bisect.bisect_right(offs, a["offset"]), "expr": " ".join(a["expr"].split()),
                     "value": a["value"], "correct": a["correct"], "repeat": a["canon"] in seen, "canon": a["canon"]})
        seen.add(a["canon"])
    out.append({"numbers": p["numbers"], "target": p["target"], "tokens": len(offs),
                "solved": check_answer(text, p["numbers"], p["target"]), "attempts": atts,
                "text_head": text[:400]})
    print(p["numbers"], p["target"], "tokens", len(offs), "attempts", len(atts), "repeats", sum(a["repeat"] for a in atts),
          "found in reasoning", any(a["correct"] for a in atts), "final answer correct", out[-1]["solved"])
json.dump(out, open("cot_trace.json", "w"))
