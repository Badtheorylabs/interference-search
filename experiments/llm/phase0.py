"""Phase 0: do independent samples waste tokens re-trying failures a sibling already hit?

Generates K independent samples per Countdown problem (batched on MLX) and saves
raw traces. Analysis lives in analyze0.py.
"""
import argparse
import json
import random
import time
from pathlib import Path

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from interference_search.countdown import check_answer, gen_hard_problem, gen_problem, prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--easy", action="store_true")
ap.add_argument("--out", default="runs/phase0.jsonl")
args = ap.parse_args()

model, tok = load(args.model)
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
rng = random.Random(args.seed)
gen = gen_problem if args.easy else gen_hard_problem
problems = [gen(rng) for _ in range(args.n)]

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {json.loads(l)["idx"] for l in out.open()}

for idx, p in enumerate(problems):
    if idx in done:
        continue
    msgs = [{"role": "user", "content": prompt_text(p)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    t0 = time.time()
    r = batch_generate(model, tok, [ids] * args.k, max_tokens=args.max_tokens,
                       sampler=sampler, verbose=False)
    dt = time.time() - t0
    samples = []
    for text in r.texts:
        samples.append({
            "text": text,
            "n_tokens": len(tok.encode(text)),
            "solved": check_answer(text, p["numbers"], p["target"]),
        })
    rec = {"idx": idx, **p, "k": args.k, "max_tokens": args.max_tokens,
           "model": args.model, "sec": round(dt, 1), "samples": samples}
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    solved = sum(s["solved"] for s in samples)
    print(f"[{idx}] {p['numbers']} -> {p['target']}  solved {solved}/{args.k}  {dt:.0f}s", flush=True)
