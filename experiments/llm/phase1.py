"""Phase 1: training-free refutation ledger on Countdown.

K streams generate in lockstep chunks. After each chunk, every full attempt a
stream wrote is evaluated exactly; wrong ones are verified refutations. Arms:

  A  independent: no notes (same chunking, so it doubles as Phase 0 data)
  P  private:     each stream is reminded of its own refuted attempts
  C  shared:      each stream is told its own AND its siblings' refuted attempts

P and C use the same note format and timing, so C - P isolates sibling coupling.
All arms reuse the same seed per problem, so streams are identical until the
first note lands. The group stops as soon as any stream writes a verified
correct expression. KV caches are carried across chunks (verified exact against
uninterrupted generation), so notes cost only their own prefill.
"""
import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from interference_search.countdown import extract_attempts, gen_hard_problem, prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
ap.add_argument("--start", type=int, default=6)
ap.add_argument("--end", type=int, default=30)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--arms", default="A,P,C")
ap.add_argument("--max-note", type=int, default=12, help="max refutations per note")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/phase1_qwen3-1.7b.jsonl")
args = ap.parse_args()

model, tok = load(args.model)
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
rng = random.Random(args.seed)
problems = [gen_hard_problem(rng) for _ in range(args.end)]  # same stream as phase0.py

NOTE_SHARED = ("\n\n[Shared scratchpad from your {n} teammates solving this same problem in parallel. "
               "These full expressions were already checked and are WRONG, so do not retry them: {items}]\n\n")
NOTE_PRIVATE = ("\n\n[Your scratchpad. You already checked these full expressions and they are WRONG, "
                "so do not retry them: {items}]\n\n")


def fmt(expr, value):
    v = int(value) if float(value).is_integer() else round(value, 3)
    return f"{' '.join(expr.split())} = {v}"


def run_arm(p, arm, seed):
    mx.random.seed(seed)
    K = args.k
    base = tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}],
                                   add_generation_prompt=True, tokenize=True)
    gen_ids = [[] for _ in range(K)]      # generated tokens only
    suffix = [list(base) for _ in range(K)]
    caches = None
    done = [False] * K
    shown = [set() for _ in range(K)]      # refutation keys already shown to each stream
    note_tokens = 0
    rounds_log = []
    found = None
    for rnd in range(args.rounds):
        active = [k for k in range(K) if not done[k]]
        if not active:
            break
        r = batch_generate(model, tok, [suffix[k] for k in active], max_tokens=args.chunk,
                           sampler=sampler, return_prompt_caches=True,
                           prompt_caches=None if caches is None else [caches[k] for k in active])
        if caches is None:
            caches = [None] * K
        for i, k in enumerate(active):
            new = tok.encode(r.texts[i]) if r.texts[i] else []
            gen_ids[k].extend(new)
            caches[k] = r.caches[i]
            if len(new) < args.chunk or not new:
                done[k] = True
        # verify attempts in every stream's generated text so far
        texts = [tok.decode(g) for g in gen_ids]
        refuted = {}  # canon -> (stream, display)
        per_stream_ref = [dict() for _ in range(K)]
        for k in range(K):
            for a in extract_attempts(texts[k], p["numbers"], p["target"]):
                if a["correct"]:
                    if found is None:
                        found = {"round": rnd, "stream": k}
                    continue
                disp = fmt(a["expr"], a["value"])
                per_stream_ref[k].setdefault(a["canon"], disp)
                refuted.setdefault(a["canon"], disp)
        gen_total = sum(len(g) for g in gen_ids)
        rounds_log.append({"round": rnd, "gen_tokens": gen_total, "distinct_refuted": len(refuted),
                           "active": len(active)})
        if found is not None:
            break
        # build notes for the next chunk
        for k in range(K):
            if done[k]:
                continue
            last = gen_ids[k][-1:]
            trim_prompt_cache(caches[k], 1)
            note = ""
            if arm in ("P", "C"):
                pool = per_stream_ref[k] if arm == "P" else refuted
                fresh = [(c, d) for c, d in pool.items() if c not in shown[k]][-args.max_note:]
                if fresh:
                    items = "; ".join(d for _, d in fresh)
                    note = (NOTE_PRIVATE.format(items=items) if arm == "P"
                            else NOTE_SHARED.format(n=K - 1, items=items))
                    shown[k].update(c for c, _ in fresh)
            note_ids = tok.encode(note) if note else []
            note_tokens += len(note_ids)
            suffix[k] = last + note_ids
    return {
        "arm": arm, "found": found is not None, "found_round": None if found is None else found["round"],
        "found_stream": None if found is None else found["stream"],
        "gen_tokens": sum(len(g) for g in gen_ids), "note_tokens": note_tokens,
        "rounds": rounds_log, "texts": [tok.decode(g) for g in gen_ids],
    }


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done_keys = set()
if out.exists():
    done_keys = {(json.loads(l)["idx"], json.loads(l)["arm"]) for l in out.open()}

for idx in range(args.start, args.end):
    p = problems[idx]
    for arm in args.arms.split(","):
        if (idx, arm) in done_keys:
            continue
        t0 = time.time()
        res = run_arm(p, arm, seed=1000 + idx)
        rec = {"idx": idx, **p, "k": args.k, "chunk": args.chunk, "rounds_max": args.rounds,
               "model": args.model, "sec": round(time.time() - t0, 1), **res}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{idx} {arm}] {p['numbers']} -> {p['target']}  found={res['found']} "
              f"round={res['found_round']} gen={res['gen_tokens']} notes={res['note_tokens']} "
              f"{rec['sec']:.0f}s", flush=True)
