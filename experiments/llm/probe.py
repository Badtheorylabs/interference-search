"""Does Qwen3-1.7B already know which Countdown states are dead, even though it can't say so?

Short answer: not beyond what plain number features carry. probe_control.py is the control that shows it.

No fine-tuning. Run the frozen model once over a state prompt, read the hidden state of the last
token at a few depths, and train a linear probe (on CPU) to predict alive/dead from exact DP labels.
Test on states from unseen problems, including a bigger size. Compare against the prompted judge
(P("Yes")) and the small refuter. Finally use the best probe as the judge in an
environment-enumerates frontier on the same 30 hard problems as llm_state.py.
"""
import json
import os
import random
import time
from functools import lru_cache

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

from interference_search.countdown import gen_hard_problem, gen_problem, moves
from interference_search.judge import load_judge

LAYERS = (14, 21, 28)
model, tok = load("mlx-community/Qwen3-1.7B-4bit")
YES = tok.encode("Yes", add_special_tokens=False)[0]
NO = tok.encode("No", add_special_tokens=False)[0]


def prompt(state, target):
    msg = (f"Countdown. Numbers: {', '.join(map(str, state))}. Target: {target}.\n"
           "Using each number exactly once with +, -, *, / (positive whole-number results only), "
           "can you reach exactly the target? Answer Yes or No.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                   tokenize=True, enable_thinking=False)


def features(items, bs=32):
    """items: list of (state, target). Returns {layer: np.array (n, d)} and prompted P(yes)."""
    feats = {l: [] for l in LAYERS}
    pyes = []
    inner = model.model
    for k in range(0, len(items), bs):
        prompts = [prompt(s, t) for s, t in items[k:k + bs]]
        L = max(len(p) for p in prompts)
        x = mx.array([p + [0] * (L - len(p)) for p in prompts])      # right padding: causal mask keeps it harmless
        last = mx.array([len(p) - 1 for p in prompts])
        h = inner.embed_tokens(x)
        mask = create_attention_mask(h, None)
        for i, layer in enumerate(inner.layers, 1):
            h = layer(h, mask, None)
            if i in LAYERS and i != len(inner.layers):
                feats[i].append(np.array(h[mx.arange(len(prompts)), last].astype(mx.float32)))
        hn = inner.norm(h)
        top = hn[mx.arange(len(prompts)), last]
        feats[len(inner.layers)].append(np.array(top.astype(mx.float32)))
        logits = model.lm_head(top) if hasattr(model, "lm_head") else inner.embed_tokens.as_linear(top)
        lg = np.array(logits.astype(mx.float32))
        pyes += list(1 / (1 + np.exp(lg[:, NO] - lg[:, YES])))
    return {l: np.concatenate(v) for l, v in feats.items()}, np.array(pyes)


def labelled_states(problems, cap=None, seed=0):
    rows = []
    for p in problems:
        t = p["target"]

        @lru_cache(maxsize=None)
        def solv(s):
            return s[0] == t if len(s) == 1 else any(solv(m) for m in moves(s))
        seen, stack = set(), [tuple(sorted(p["numbers"]))]
        while stack:
            s = stack.pop()
            if s in seen or len(s) == 1:
                continue
            seen.add(s)
            stack.extend(moves(s))
        rows += [(s, t, solv(s)) for s in sorted(seen)]
    if cap and len(rows) > cap:
        rows = random.Random(seed).sample(rows, cap)
    return rows


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_probe(X, y, Xv, yv):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Vt = torch.tensor((Xv - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    best = (-1, None)
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=wd)
        pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))
        for _ in range(300):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = auc((Vt @ w + b).numpy(), yv)
        if a > best[0]:
            best = (a, (w.detach().clone(), b.detach().clone(), wd))
    w, b, wd = best[1]
    return (lambda Z: torch.sigmoid(torch.tensor((Z - mu) / sd, dtype=torch.float32) @ w + b).numpy()), wd, best[0]


t0 = time.time()
rng = random.Random(7)
train_p = [gen_hard_problem(rng) for _ in range(70)]
val_p, train_p = train_p[:10], train_p[10:]
test4_p = [gen_hard_problem(random.Random(900 + i)) for i in range(30)]
test5_p = [gen_problem(random.Random(1900 + i), n_numbers=5) for i in range(8)]

train = labelled_states(train_p)
alive = [r for r in train if r[2]]
dead = [r for r in train if not r[2]]
train = alive + random.Random(0).sample(dead, min(len(dead), 3 * len(alive)))   # 1:3 balance
val = labelled_states(val_p)
test4 = labelled_states(test4_p)
test5 = labelled_states(test5_p, cap=2000)
print(f"states: train {len(train)} ({len(alive)} alive), val {len(val)}, test4 {len(test4)}, test5 {len(test5)}"
      f"  ({time.time() - t0:.0f}s)", flush=True)

F = {}
for name, rows in (("train", train), ("val", val), ("test4", test4), ("test5", test5)):
    F[name] = features([(s, t) for s, t, _ in rows])
    print(f"features {name} done ({time.time() - t0:.0f}s)", flush=True)
Y = {n: np.array([r[2] for r in rows], dtype=float) for n, rows in
     (("train", train), ("val", val), ("test4", test4), ("test5", test5))}

# the trained Countdown judge on the same states
_judge = load_judge()


def ref_scores(rows):
    return __import__("numpy").array(_judge([s for s, _, _ in rows], [t for _, t, _ in rows]) if rows else [])


report = {}
print("\nAUC (alive vs dead) on states from unseen problems")
print("judge                          4-number   5-number (bigger)")
report["prompted P(yes)"] = (auc(F["test4"][1], Y["test4"]), auc(F["test5"][1], Y["test5"]))
report["small refuter"] = (auc(ref_scores(test4), Y["test4"]), auc(ref_scores(test5), Y["test5"]))
probes = {}
for l in LAYERS:
    fn, wd, va = train_probe(F["train"][0][l], Y["train"], F["val"][0][l], Y["val"])
    probes[l] = fn
    report[f"probe layer {l}"] = (auc(fn(F["test4"][0][l]), Y["test4"]), auc(fn(F["test5"][0][l]), Y["test5"]))
for k, (a4, a5) in report.items():
    print(f"  {k:28s}  {a4:8.3f}   {a5:8.3f}")
best_layer = max(LAYERS, key=lambda l: report[f"probe layer {l}"][0])
print(f"best probe layer {best_layer}  ({time.time() - t0:.0f}s)", flush=True)

# use the best probe as a judge in an environment-enumerates frontier on llm_state.py's 30 hard problems
# the same 30 hard problems as llm_state.py (seed 11)
rng11 = random.Random(11)
problems = [gen_hard_problem(rng11) for _ in range(30)]


def children(state):
    return sorted(set(moves(state)))


def enum_frontier(p, judge, width=6, max_evals=150):
    target, live, evals, rounds = p["target"], [tuple(sorted(p["numbers"]))], 0, 0
    while live and evals < max_evals:
        rounds += 1
        kids = sorted({c for s in live for c in children(s)})
        if (target,) in kids:
            return True, evals, rounds
        kids = [c for c in kids if len(c) > 1][: max_evals - evals]
        if not kids:
            break
        sc = judge(kids, target)
        evals += len(kids)
        live = [kids[j] for j in np.argsort(-np.asarray(sc))[:width]]
    return False, evals, rounds


def probe_judge(kids, target):
    feats, _ = features([(k, target) for k in kids])
    return probes[best_layer](feats[best_layer])


def pyes_judge(kids, target):
    return features([(k, target) for k in kids])[1]


def ref_judge(kids, target):
    return ref_scores([(k, target, None) for k in kids])


print("\nenvironment enumerates, judge ranks (30 hard 4-number problems, <= 150 judged states each)")
for name, j in (("prompted P(yes)", pyes_judge), (f"probe layer {best_layer}", probe_judge), ("small refuter", ref_judge)):
    res = [enum_frontier(p, j) for p in problems]
    print(f"  {name:22s} solved {sum(r[0] for r in res)}/30   mean judged states {np.mean([r[1] for r in res]):.0f}", flush=True)
    report[f"search {name}"] = sum(r[0] for r in res) / 30
os.makedirs("runs", exist_ok=True)
json.dump(report, open("runs/probe_results.json", "w"), indent=1)
print(f"done ({time.time() - t0:.0f}s)")
