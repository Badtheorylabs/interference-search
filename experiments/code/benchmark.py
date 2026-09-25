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
import random
import time
from collections import defaultdict

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from pathlib import Path

from interference_search.program_domain import Code, State, execute, extract_code, feedback

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--screen", type=int, default=150)
ap.add_argument("--budget", type=int, default=1500)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--arms", default="best-of-N,transcript,linear state,Interference Search")
args = ap.parse_args()

model, tok = load("mlx-community/Qwen3-1.7B-4bit")
dom = Code(model, tok)
data = json.load(open(ROOT / "data" / "mbpp_sanitized.json"))
t0 = time.time()

# screen: keep problems the model gets wrong on its first greedy attempt (cached)
import os
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


import mlx.core as mx
mx.set_cache_limit(2 * 1024 ** 3)   # stop MLX's buffer cache from growing across hundreds of calls


def gen(prompts, token_budget=12_000, max_bs=24):
    """Generate in batches sized by total tokens (prompt + completion), so the attention cache stays
    bounded even when transcript prompts grow long."""
    texts, k = [], 0
    while k < len(prompts):
        bs = 1
        while k + bs < len(prompts) and bs < max_bs and \
                (bs + 1) * (max(len(p) for p in prompts[k:k + bs + 1]) + dom.max_tokens) <= token_budget:
            bs += 1
        r = batch_generate(model, tok, prompts[k:k + bs], max_tokens=dom.max_tokens, sampler=dom.sampler,
                           verbose=False)
        texts += r.texts
        k += bs
        mx.clear_cache()
    return texts, [len(tok.encode(t)) for t in texts]


def run(arm):
    P = len(problems)
    st = [{"solved": False, "gen": 0, "prompt": 0, "rounds": 0, "live": [State()], "seen": set(),
           "chat": None, "merged": 0, "raw": 0} for _ in range(P)]
    while True:
        act = [i for i in range(P) if not st[i]["solved"] and st[i]["gen"] < args.budget]
        if not act:
            break
        jobs = []                                    # (problem, parent, prompt)
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
            else:
                for pr, j in dom.expand_prompts(s["live"], p):
                    jobs.append((i, s["live"][j], pr))
        texts, counts = gen([j[2] for j in jobs])
        pools = defaultdict(dict)
        for (i, parent, pr), t, c in zip(jobs, texts, counts):
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
            elif arm == "Interference Search":
                s["raw"] += 1
                k = dom.key(new)
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
        if arm == "Interference Search":
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
                ranked = sorted(pool.values(), key=lambda v: (-v[0].passed, -v[1]))
                s["live"] = [v[0] for v in ranked[:args.width]]
    n = len(problems)
    solved = [s for s in st if s["solved"]]
    return {"arm": arm, "solved": len(solved), "of": n,
            "gen_tokens_mean": sum(s["gen"] for s in st) / n,
            "gen_tokens_when_solved": sum(s["gen"] for s in solved) / max(len(solved), 1),
            "prompt_tokens_mean": sum(s["prompt"] for s in st) / n,
            "rounds_when_solved": sum(s["rounds"] for s in solved) / max(len(solved), 1),
            "merge_rate": (sum(s["merged"] for s in st) / max(sum(s["raw"] for s in st), 1)) if arm == "Interference Search" else None,
            "sec": round(time.time() - t0)}


done = {}
if os.path.exists("bench_code_partial.json"):
    done = {r["arm"]: r for r in json.load(open("bench_code_partial.json"))}
results = []
for arm in args.arms.split(","):
    res = done.get(arm) or run(arm)
    results.append(res)
    json.dump(results, open("bench_code_partial.json", "w"), indent=1)
    print(json.dumps(res), flush=True)
json.dump({"args": vars(args), "results": results, "problem_ids": [p["task_id"] for p in problems]},
          open("code_benchmark.json", "w"), indent=1)
print(f"\n{len(problems)} MBPP problems Qwen3-1.7B fails on its first greedy try; {args.budget} generated tokens each")
print("arm                    solved   gen tokens (mean)   prompt tokens (mean)   rounds when solved")
for r in results:
    print(f"{r['arm']:22s}  {r['solved']:2d}/{r['of']}   {r['gen_tokens_mean']:12.0f}   {r['prompt_tokens_mean']:18.0f}   {r['rounds_when_solved']:10.1f}")
