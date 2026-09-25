"""Transcript vs explicit state vs shared state frontier, on a real LLM (Qwen3-1.7B, MLX), no training.

Every arm gets the same budget of generated tokens per problem. Calls are batched across problems.

  transcript   normal chain-of-thought with thinking on: one growing transcript per problem
  state        Markov: each step the model sees only the current numbers and target and proposes one
               move; the environment applies it; dead end -> restart from the start numbers
  frontier     each live state asks for up to 3 candidate moves; the environment applies them, merges
               identical states (a state proposed by several parents gets their votes), keeps the best
               `width` states, and all advance one level together
  frontier+ref same, ranked by the trained Countdown judge instead of votes

Reported per arm: solve rate, generated tokens, prompt tokens, sequential LLM rounds.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from fractions import Fraction

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from interference_search.countdown import check_answer, gen_hard_problem, prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--budget", type=int, default=1500, help="generated tokens per problem")
ap.add_argument("--width", type=int, default=6)
ap.add_argument("--max-evals", type=int, default=150, help="states a judge may score per problem (enum arms)")
ap.add_argument("--arms", default="transcript,state,frontier,frontier+ref")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="runs/llm_state.jsonl")
args = ap.parse_args()

model, tok = load(args.model)
sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20)
rng = random.Random(args.seed)
problems = [gen_hard_problem(rng) for _ in range(args.n)]

MOVE_RE = re.compile(r"(\d+)\s*([+\-*/x×÷])\s*(\d+)")


PROMPT_RNG = random.Random(123)


def state_prompt(nums, target, k):
    nums = list(nums)
    PROMPT_RNG.shuffle(nums)  # the model tends to combine the first two listed numbers
    ask = "the single best move" if k == 1 else f"up to {k} different candidate moves, best first, one per line"
    msg = (f"Countdown. Numbers: {', '.join(map(str, nums))}. Target: {target}.\n"
           "A move picks two of the numbers and one operation (+, -, *, /). The result replaces both. "
           "Results must be positive whole numbers. You win when a single number equal to the target remains.\n"
           f"Reply with {ask}, in the form `a op b`, and nothing else.")
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                  tokenize=True, enable_thinking=False)
    # prefill the start of the answer so the first generated tokens are a move, not an essay
    return ids + tok.encode("Move:\n1. " if k == 1 else "Candidate moves:\n1. ", add_special_tokens=False)


def apply_move(state, a, op, b):
    s = list(state)
    if a not in s:
        return None
    s.remove(a)
    if b not in s:
        return None
    s.remove(b)
    op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
    if op == "+":
        r = a + b
    elif op == "*":
        r = a * b
    elif op == "-":
        r = a - b
    else:
        if b == 0 or a % b:
            return None
        r = a // b
    if r <= 0:
        return None
    return tuple(sorted(s + [r]))


def parse_moves(text, state, k):
    text = "1. " + re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = []
    for m in MOVE_RE.finditer(text):
        nxt = apply_move(state, int(m.group(1)), m.group(2), int(m.group(3)))
        if nxt is not None and nxt not in out:
            out.append(nxt)
        if len(out) >= k:
            break
    return out


def generate(prompts, max_tokens):
    """Returns texts, generated-token counts, prompt-token total."""
    if not prompts:
        return [], [], 0
    r = batch_generate(model, tok, prompts, max_tokens=max_tokens, sampler=sampler, verbose=False)
    counts = [len(tok.encode(t)) for t in r.texts]
    return r.texts, counts, sum(len(p) for p in prompts)


# ---------- arms ----------

def run_transcript():
    prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}], add_generation_prompt=True,
                                       tokenize=True) for p in problems]
    texts, counts, ptoks = generate(prompts, args.budget)
    return [{"solved": check_answer(t, p["numbers"], p["target"]), "gen": c, "prompt": len(pr), "rounds": 1,
             "seq_tokens": c} for t, c, p, pr in zip(texts, counts, problems, prompts)]


def legal_children(state):
    out = set()
    for i in range(len(state)):
        for j in range(len(state)):
            if i != j:
                for op in "+-*/":
                    n = apply_move(state, state[i], op, state[j])
                    if n is not None:
                        out.add(n)
    return sorted(out)


def run_state():
    """Markov state, plus memory: at each state, children already tried are never chosen again."""
    st = [{"state": tuple(sorted(p["numbers"])), "solved": False, "gen": 0, "prompt": 0, "rounds": 0,
           "tried": defaultdict(set), "dead": set(), "path": []} for p in problems]
    while True:
        act = [i for i, s in enumerate(st) if not s["solved"] and s["gen"] < args.budget]
        if not act:
            break
        prompts = [state_prompt(st[i]["state"], problems[i]["target"], 3) for i in act]
        texts, counts, _ = generate(prompts, 45)
        for i, t, c, pr in zip(act, texts, counts, prompts):
            s = st[i]
            s["gen"] += c; s["prompt"] += len(pr); s["rounds"] += 1
            cur, target = s["state"], problems[i]["target"]
            start = tuple(sorted(problems[i]["numbers"]))
            fresh = [n for n in parse_moves(t, cur, 3) if n not in s["tried"][cur] and n not in s["dead"]]
            if not fresh:  # model only proposed tried moves: take an untried legal move instead
                fresh = [n for n in legal_children(cur) if n not in s["tried"][cur] and n not in s["dead"]]
            if not fresh:  # every move from here has been tried: this state is exhausted
                s["dead"].add(cur)
                s["state"], s["path"] = start, []
                continue
            n = fresh[0]
            s["tried"][cur].add(n)
            if n == (target,):
                s["solved"] = True
            elif len(n) == 1:
                s["state"], s["path"] = start, []
            else:
                s["path"].append(cur)
                s["state"] = n
    return [{"solved": s["solved"], "gen": s["gen"], "prompt": s["prompt"], "rounds": s["rounds"],
             "seq_tokens": s["gen"]} for s in st]


_judge = None


def refuter_scores(states, target):
    """Scores from the trained Countdown judge (loaded on first use)."""
    global _judge
    if _judge is None:
        from interference_search.judge import load_judge
        _judge = load_judge()
    return _judge(list(states), target)


def run_frontier(use_ref):
    st = [{"live": [tuple(sorted(p["numbers"]))], "solved": False, "gen": 0, "prompt": 0, "rounds": 0,
           "seq_tokens": 0, "expanded": set()} for p in problems]
    while True:
        jobs = [(i, s) for i, s in enumerate(st) if not st[i]["solved"] and st[i]["gen"] < args.budget
                for s in st[i]["live"]]
        if not jobs:
            break
        prompts = [state_prompt(s, problems[i]["target"], 3) for i, s in jobs]
        texts, counts, _ = generate(prompts, 45)
        votes = defaultdict(lambda: defaultdict(float))
        round_max = defaultdict(int)
        for (i, s), t, c, pr in zip(jobs, texts, counts, prompts):
            p = st[i]
            p["expanded"].add(s)
            p["gen"] += c; p["prompt"] += len(pr)
            round_max[i] = max(round_max[i], c)
            for rank, n in enumerate(parse_moves(t, s, 3)):
                votes[i][n] += 1.0 / (rank + 1)      # identical next states merge and pool their votes
        for i in {j for j, _ in jobs}:
            p = st[i]
            p["rounds"] += 1
            p["seq_tokens"] += round_max[i]
            target = problems[i]["target"]
            cand = votes[i]
            if (target,) in cand:
                p["solved"] = True
                continue
            cand = {n: v for n, v in cand.items() if len(n) > 1 and n not in p["expanded"]}
            if not cand:
                p["live"] = [tuple(sorted(problems[i]["numbers"]))]   # frontier died: restart
                continue
            keys = list(cand)
            if use_ref:
                sc = refuter_scores(keys, target)
                order = sorted(range(len(keys)), key=lambda j: -sc[j])
            else:
                order = sorted(range(len(keys)), key=lambda j: -cand[keys[j]])
            p["live"] = [keys[j] for j in order[:args.width]]
    return [{"solved": s["solved"], "gen": s["gen"], "prompt": s["prompt"], "rounds": s["rounds"],
             "seq_tokens": s["seq_tokens"]} for s in st]


YES = tok.encode("Yes", add_special_tokens=False)[0]
NO = tok.encode("No", add_special_tokens=False)[0]


def judge_prompt(nums, target):
    nums = list(nums)
    PROMPT_RNG.shuffle(nums)
    msg = (f"Countdown. Numbers: {', '.join(map(str, nums))}. Target: {target}.\n"
           "Using each number exactly once with +, -, *, / (positive whole-number results only), "
           "can you reach exactly the target? Answer Yes or No.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                   tokenize=True, enable_thinking=False)


def llm_judge(states, target, bs=48):
    """P(yes) from one forward pass per state, batched with right padding."""
    import mlx.core as mx
    out = []
    for k in range(0, len(states), bs):
        prompts = [judge_prompt(s, target) for s in states[k:k + bs]]
        L = max(len(p) for p in prompts)
        arr = mx.array([p + [0] * (L - len(p)) for p in prompts])
        logits = model(arr)
        for j, p in enumerate(prompts):
            v = logits[j, len(p) - 1]
            y, n = v[YES].item(), v[NO].item()
            out.append(1 / (1 + math.exp(n - y)))
    return out, sum(len(p) for p in prompts)


def run_enum(judge):
    """Level-synchronous frontier: the environment lists every legal child, the judge ranks them."""
    res = []
    for p in problems:
        target, live = p["target"], [tuple(sorted(p["numbers"]))]
        evals, rounds, ptoks, solved = 0, 0, 0, False
        while live and evals < args.max_evals:
            rounds += 1
            kids = sorted({c for s in live for c in legal_children(s)})
            if (target,) in kids:
                solved = True
                break
            kids = [c for c in kids if len(c) > 1][: args.max_evals - evals]
            if not kids:
                break
            if judge == "llm":
                sc, pt = llm_judge(kids, target)
                ptoks += sum(len(judge_prompt(c, target)) for c in kids)
            else:
                sc = refuter_scores(kids, target)
            evals += len(kids)
            order = sorted(range(len(kids)), key=lambda j: -sc[j])
            live = [kids[j] for j in order[:args.width]]
        res.append({"solved": solved, "gen": evals if judge == "llm" else 0, "prompt": ptoks,
                    "rounds": rounds, "seq_tokens": rounds, "evals": evals})
    return res


ARMS = {"transcript": run_transcript, "state": run_state, "frontier": lambda: run_frontier(False),
        "frontier+ref": lambda: run_frontier(True),
        "enum+llm": lambda: run_enum("llm"), "enum+ref": lambda: run_enum("ref")}

results = {}
for arm in args.arms.split(","):
    t0 = time.time()
    res = ARMS[arm]()
    n = len(res)
    solved = [r for r in res if r["solved"]]
    summary = {"arm": arm, "solved": len(solved) / n,
               "gen_tokens": sum(r["gen"] for r in res) / n,
               "gen_tokens_when_solved": sum(r["gen"] for r in solved) / max(len(solved), 1),
               "prompt_tokens": sum(r["prompt"] for r in res) / n,
               "rounds_when_solved": sum(r["rounds"] for r in solved) / max(len(solved), 1),
               "sequential_tokens_when_solved": sum(r["seq_tokens"] for r in solved) / max(len(solved), 1),
               "sec": round(time.time() - t0)}
    results[arm] = {"summary": summary, "per_problem": res}
    print(json.dumps(summary), flush=True)

with open(args.out, "a") as f:
    f.write(json.dumps({"args": vars(args), "problems": problems, "results": results}) + "\n")
print()
print(f"{args.n} hard 4-number problems, {args.budget} generated tokens per problem")
print("arm            solved   gen tokens   gen when solved   prompt tokens   rounds when solved   sequential tokens")
for arm, r in results.items():
    s = r["summary"]
    print(f"{arm:13s}  {s['solved']:6.0%}   {s['gen_tokens']:10.0f}   {s['gen_tokens_when_solved']:15.0f}   "
          f"{s['prompt_tokens']:13.0f}   {s['rounds_when_solved']:18.1f}   {s['sequential_tokens_when_solved']:17.0f}")
