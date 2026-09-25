"""Phase 1b: structural suppression by rewind-and-resample.

Phase 1 showed that telling a small model "these are wrong, do not retry" primes
the listed expressions. Here suppression happens in decoding instead: when a
stream writes a full attempt that is already refuted, the stream is rewound to
the start of that line and resampled. Discarded tokens are charged to the
stream's budget, so every arm spends the same generated-token budget.

  A       independent, no rewinds
  Rself   rewind on repeats of the stream's own refuted attempts
  Rall    rewind on repeats of its own OR any sibling's refuted attempts
          (siblings' refutations become visible at chunk boundaries, lockstep)

Rall - Rself isolates the value of coupling streams together.
"""
import argparse
import json
import random
import time
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from interference_search.countdown import OPS, _combine, extract_attempts, gen_problem, prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--numbers", type=int, default=5)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--budget", type=int, default=2048, help="generated tokens per stream, discarded ones included")
ap.add_argument("--arms", default="A,Rself,Rall")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="runs/phase1b_qwen3-1.7b.jsonl")
args = ap.parse_args()

model, tok = load(args.model)
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
eos = set(tok.eos_token_ids)


def gen_hard(rng):
    """n-number instance with no pairwise anchor within 25 of the target."""
    while True:
        p = gen_problem(rng, n_numbers=args.numbers)
        nums, t = p["numbers"], p["target"]
        vals = [_combine(Fraction(a), Fraction(b), op) for i, a in enumerate(nums)
                for j, b in enumerate(nums) if i != j for op in OPS]
        if not any(v is not None and abs(v - t) <= 25 for v in vals):
            return p


rng = random.Random(args.seed)
problems = [gen_hard(rng) for _ in range(args.n)]


def generate_chunk(prompts, caches, max_tokens):
    """Like mlx_lm.batch_generate, but returns the sampled token ids and caches."""
    gen = BatchGenerator(model, stop_tokens=[[t] for t in eos])
    uids = gen.insert(prompts, max_tokens, caches=caches)
    toks = {u: [] for u in uids}
    out_caches, stopped = {}, {u: False for u in uids}
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            else:
                stopped[r.uid] = True
            if r.finish_reason is not None:
                out_caches[r.uid] = r.prompt_cache
    gen.close()
    return [toks[u] for u in uids], [out_caches[u] for u in uids], [stopped[u] for u in uids]


def char_len(ids, i):
    return len(tok.decode(ids[:i]))


def tokens_before_char(ids, c):
    """Largest i such that decode(ids[:i]) fits within the first c characters."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if char_len(ids, mid) <= c:
            lo = mid
        else:
            hi = mid - 1
    return lo


def run_arm(p, arm, seed):
    mx.random.seed(seed)
    K = args.k
    base = tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}],
                                   add_generation_prompt=True, tokenize=True)
    ids = [[] for _ in range(K)]            # kept generated tokens
    caches = [None] * K                     # cache covers base + ids (all of them)
    spent = [0] * K
    done = [False] * K
    own = [dict() for _ in range(K)]        # canon -> True, refuted by this stream (kept text)
    known = set()                           # all refutations visible at this round's start
    rewinds, discarded = 0, 0
    found, rounds = None, []
    rnd = 0
    while True:
        active = [k for k in range(K) if not done[k] and spent[k] < args.budget]
        if not active:
            break
        prompts, cs, starts = [], [], {}
        for k in active:
            starts[k] = len(ids[k])
            if caches[k] is None:
                prompts.append(list(base)); cs.append(None)
            else:
                full = base + ids[k]
                trim_prompt_cache(caches[k], 1)
                prompts.append(full[-1:]); cs.append(caches[k])
        maxes = [min(args.chunk, args.budget - spent[k]) for k in active]
        _t = time.time()
        new_toks, new_caches, stopped = generate_chunk(prompts, cs, maxes)
        print(f"   chunk {rnd} {arm} streams={len(active)} {time.time() - _t:.1f}s", flush=True)
        for i, k in enumerate(active):
            ids[k].extend(new_toks[i])
            caches[k] = new_caches[i]
            spent[k] += len(new_toks[i])
            if stopped[i] or not new_toks[i]:
                done[k] = True
            extra = caches[k][0].offset - (len(base) + len(ids[k]))
            if extra > 0:  # e.g. an EOS token that is in the cache but not in ids
                trim_prompt_cache(caches[k], extra)
            assert caches[k][0].offset == len(base) + len(ids[k]), (caches[k][0].offset, len(base), len(ids[k]))
        # verify, detect repeats, rewind
        round_new = set()
        for k in active:
            text = tok.decode(ids[k])
            start_char = char_len(ids[k], starts[k])
            atts = extract_attempts(text, p["numbers"], p["target"])
            if any(a["correct"] for a in atts) and found is None:
                found = {"round": rnd, "stream": k}
            if found is not None:
                continue
            seen_self = set()
            offender = None
            for a in atts:
                c = a["canon"]
                if a["offset"] < start_char:
                    seen_self.add(c)
                    continue
                repeat_self = c in seen_self
                repeat_sib = arm == "Rall" and c in known and c not in own[k]
                if arm != "A" and (repeat_self or repeat_sib):
                    offender = a
                    break
                seen_self.add(c)
            if offender is not None:
                line_start = text.rfind("\n", 0, offender["offset"]) + 1
                keep = tokens_before_char(ids[k], line_start)
                drop = len(ids[k]) - keep
                if drop > 0:
                    trim_prompt_cache(caches[k], drop)
                    ids[k] = ids[k][:keep]
                    assert caches[k][0].offset == len(base) + len(ids[k])
                    discarded += drop
                    rewinds += 1
                    done[k] = False  # a stream that hit EOS after the offender gets to continue
                    text = tok.decode(ids[k])
                    atts = extract_attempts(text, p["numbers"], p["target"])
            for a in atts:
                if not a["correct"]:
                    own[k][a["canon"]] = True
                    round_new.add(a["canon"])
        known |= round_new
        rounds.append({"round": rnd, "spent": sum(spent), "kept": sum(len(x) for x in ids),
                       "known": len(known), "rewinds": rewinds})
        if found is not None:
            break
        rnd += 1
    return {"arm": arm, "found": found is not None,
            "found_round": None if found is None else found["round"],
            "spent_at_end": sum(spent), "rewinds": rewinds, "discarded": discarded,
            "rounds": rounds, "texts": [tok.decode(x) for x in ids]}


out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done_keys = set()
if out.exists():
    done_keys = {(json.loads(l)["idx"], json.loads(l)["arm"]) for l in out.open()}

for idx, p in enumerate(problems):
    for arm in args.arms.split(","):
        if (idx, arm) in done_keys:
            continue
        t0 = time.time()
        res = run_arm(p, arm, seed=5000 + idx)
        rec = {"idx": idx, **p, "k": args.k, "chunk": args.chunk, "budget": args.budget,
               "model": args.model, "sec": round(time.time() - t0, 1), **res}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{idx} {arm}] {p['numbers']} -> {p['target']}  found={res['found']} round={res['found_round']} "
              f"spent={res['spent_at_end']} rewinds={res['rewinds']} discarded={res['discarded']} "
              f"{rec['sec']:.0f}s", flush=True)
