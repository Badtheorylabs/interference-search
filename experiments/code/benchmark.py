"""Code domain benchmark: Qwen3-1.7B on MBPP problems it fails on the first greedy try.

Arms, all at the same generated-token budget per problem, all batched across problems each round:
  best-of-N         fresh programs only; solved if any passes every test (a strong pass@N baseline)
  transcript        the usual agent loop: one chat that grows with every attempt and its test feedback
  linear state      one program at a time, revised from (current program + its test feedback) only
  Interference      frontier of programs; each round every live program gets k revisions; programs with
  Search            identical behaviour on the tests merge; rank by tests passed (votes break ties);
                    keep the best `width`
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from interference_search.program_domain import Code, State, execute, extract_code, feedback
from interference_search.frontier_v2 import Candidate, select_frontier

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--screen", type=int, default=150)
ap.add_argument("--budget", type=int, default=1500)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--arms", default="best-of-N,transcript,linear state,Interference Search")
ap.add_argument("--out", default="bench_code_partial.json")
ap.add_argument("--seed", type=int, default=None, help="reset sampling to this seed for each arm")
args = ap.parse_args()

model, tok = load("mlx-community/Qwen3-1.7B-4bit")
dom = Code(model, tok)
data = json.load(open(ROOT / "data" / "mbpp_sanitized.json"))
t0 = time.time()

# screen: keep problems the model gets wrong on its first greedy attempt (cached)
cand = data[:args.screen]
cache = ROOT / "results" / "code" / f"screen_{args.screen}.json"
if os.path.exists(cache):
    first_try = json.load(open(cache))
else:
    greedy = make_sampler(temp=0.0)
    texts = []
    for k in range(0, len(cand), 32):
        texts += batch_generate(model, tok, [dom.fresh_prompt(p) for p in cand[k:k + 32]], max_tokens=dom.max_tokens,
                                sampler=greedy, verbose=False).texts
    first_try = [dom.is_goal(State(extract_code(t), execute(extract_code(t), p["test_list"])), p)
                 for t, p in zip(texts, cand)]
    json.dump(first_try, open(cache, "w"))
failed = [p for p, ok in zip(cand, first_try) if not ok]
print(f"screen: greedy first try solves {sum(first_try)}/{len(cand)}; {len(failed)} failures, using {min(args.n, len(failed))}"
      f"  ({time.time() - t0:.0f}s)", flush=True)
problems = random.Random(0).sample(failed, min(args.n, len(failed)))

mx.set_cache_limit(2 * 1024 ** 3)   # stop MLX's buffer cache from growing across hundreds of calls


def gen(prompts, token_budget=12_000, max_bs=24, max_new_tokens=None):
    """Generate in batches sized by total tokens (prompt + completion), so the attention cache stays
    bounded even when transcript prompts grow long."""
    max_new_tokens = dom.max_tokens if max_new_tokens is None else max_new_tokens
    texts, k = [], 0
    while k < len(prompts):
        bs = 1
        while k + bs < len(prompts) and bs < max_bs and \
                (bs + 1) * (max(len(p) for p in prompts[k:k + bs + 1]) + max_new_tokens) <= token_budget:
            bs += 1
        r = batch_generate(model, tok, prompts[k:k + bs], max_tokens=max_new_tokens, sampler=dom.sampler,
                           verbose=False)
        texts += r.texts
        k += bs
        mx.clear_cache()
    return texts, [len(tok.encode(t)) for t in texts]


def run(arm):
    run_t0 = time.time()
    if args.seed is not None:
        mx.random.seed(args.seed)
    P = len(problems)
    st = [{"solved": False, "gen": 0, "prompt": 0, "rounds": 0, "live": [State()], "seen": set(),
           "chat": None, "merged": 0, "raw": 0, "signature_collisions": 0} for _ in range(P)]
    while True:
        act = [i for i in range(P) if not st[i]["solved"] and st[i]["gen"] < args.budget]
        if not act:
            break
        jobs = []                                    # (problem, parent, prompt[, completion cap])
        for i in act:
            p, s = problems[i], st[i]
            if arm == "best-of-N":
                jobs += [(i, None, dom.fresh_prompt(p))] * dom.fresh_k
            elif arm == "transcript":
                if s["chat"] is None:
                    s["chat"] = [{"role": "user", "content": f"{p['prompt']}\nYour function must pass this test:\n"
                                  f"{p['test_list'][0]}\nReply with only the Python code in one ```python block."}]
                jobs.append((i, None, tok.apply_chat_template(s["chat"], add_generation_prompt=True, tokenize=True,
                                                              enable_thinking=False)))
            elif arm == "linear state":
                cur = s["live"][0]
                jobs.append((i, cur, dom.fresh_prompt(p) if cur.src is None else dom.revise_prompt(p, cur)))
            elif arm == "Interference Search v2":
                remaining = args.budget - s["gen"]
                for pr, j in dom.expand_prompts(s["live"], p):
                    if remaining <= 0:
                        break
                    cap = min(dom.max_tokens, remaining)
                    jobs.append((i, s["live"][j], pr, cap))
                    remaining -= cap
            else:
                for pr, j in dom.expand_prompts(s["live"], p):
                    jobs.append((i, s["live"][j], pr))
        if arm == "Interference Search v2":
            texts, counts = [None] * len(jobs), [0] * len(jobs)
            for cap in sorted({job[3] for job in jobs}, reverse=True):
                indices = [k for k, job in enumerate(jobs) if job[3] == cap]
                part_texts, part_counts = gen([jobs[k][2] for k in indices], max_new_tokens=cap)
                for k, generated, count in zip(indices, part_texts, part_counts):
                    texts[k], counts[k] = generated, count
        else:
            texts, counts = gen([j[2] for j in jobs])
        pools = defaultdict(dict)
        for job, t, c in zip(jobs, texts, counts):
            i, parent, pr = job[:3]
            s, p = st[i], problems[i]
            s["gen"] += c
            s["prompt"] += len(pr)
            new = dom.to_states([t], p)[0]
            if dom.is_goal(new, p) and s["gen"] <= args.budget:   # only count solves within the budget
                s["solved"] = True
            if arm == "transcript":
                s["chat"] += [{"role": "assistant", "content": f"```python\n{new.src}\n```"},
                              {"role": "user", "content": f"Test results:\n{feedback(p, new)}\n"
                               "Fix the code so every test passes. Reply with only the corrected Python code."}]
            elif arm == "linear state":
                s["live"] = [new]
            elif arm in ("Interference Search", "Interference Search v2"):
                s["raw"] += 1
                k = dom.key(new) if arm == "Interference Search" else dom.exact_key(new)
                if k in s["seen"]:
                    s["merged"] += 1
                    continue
                if k in pools[i]:
                    pools[i][k][1] += 1
                    s["merged"] += 1
                else:
                    pools[i][k] = [new, 1]
        for i in act:
            st[i]["rounds"] += 1
        print(f"   [{arm}] round {st[act[0]]['rounds']}: {len(jobs)} programs, active {len(act)}, "
              f"solved so far {sum(x['solved'] for x in st)}/{P}, mean tokens {sum(x['gen'] for x in st) / P:.0f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if arm in ("Interference Search", "Interference Search v2"):
            for i in act:
                s = st[i]
                if s["solved"]:
                    continue
                pool = pools[i]
                for k in pool:
                    s["seen"].add(k)
                if not pool:
                    s["live"] = [State()]                # everything merged into known behaviour: start fresh
                    continue
                if arm == "Interference Search":
                    ranked = sorted(pool.values(), key=lambda v: (-v[0].passed, -v[1]))
                    s["live"] = [v[0] for v in ranked[:args.width]]
                else:
                    selection = select_frontier(
                        (Candidate(state=v[0], exact_key=k, signature=dom.signature(v[0]),
                                   score=v[0].passed, support=v[1]) for k, v in pool.items()),
                        width=args.width,
                    )
                    s["signature_collisions"] += selection.signature_collisions
                    s["live"] = [candidate.state for candidate in selection.live]
    n = len(problems)
    solved = [s for s in st if s["solved"]]
    return {"arm": arm, "solved": len(solved), "of": n,
            "gen_tokens_mean": sum(s["gen"] for s in st) / n,
            "gen_tokens_when_solved": sum(s["gen"] for s in solved) / max(len(solved), 1),
            "prompt_tokens_mean": sum(s["prompt"] for s in st) / n,
            "rounds_when_solved": sum(s["rounds"] for s in solved) / max(len(solved), 1),
            "merge_rate": (sum(s["merged"] for s in st) / max(sum(s["raw"] for s in st), 1))
            if arm in ("Interference Search", "Interference Search v2") else None,
            "signature_collisions": sum(s["signature_collisions"] for s in st)
            if arm == "Interference Search v2" else None,
            "sec": round(time.time() - run_t0)}


done = {}
meta_path = Path(args.out + ".meta.json")
meta = {"args": vars(args), "problem_ids": [p["task_id"] for p in problems],
        "model": "mlx-community/Qwen3-1.7B-4bit", "max_tokens_per_call": dom.max_tokens}
if os.path.exists(args.out) and meta_path.exists() and json.load(open(meta_path)) == meta:
    done = {r["arm"]: r for r in json.load(open(args.out))}
results = []
for arm in args.arms.split(","):
    res = done.get(arm) or run(arm)
    results.append(res)
    json.dump(results, open(args.out, "w"), indent=1)
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(json.dumps(res), flush=True)
json.dump({"args": vars(args), "results": results, "problem_ids": [p["task_id"] for p in problems]},
          open("code_benchmark.json", "w"), indent=1)
print(f"\n{len(problems)} MBPP problems Qwen3-1.7B fails on its first greedy try; {args.budget} generated tokens each")
print("arm                    solved   gen tokens (mean)   prompt tokens (mean)   rounds when solved")
for r in results:
    print(f"{r['arm']:22s}  {r['solved']:2d}/{r['of']}   {r['gen_tokens_mean']:12.0f}   {r['prompt_tokens_mean']:18.0f}   {r['rounds_when_solved']:10.1f}")
