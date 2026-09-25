"""The control for probe.py: a linear probe on simple number features, with the same training states.

If this probe judges states as well as the probe on Qwen3-1.7B's hidden states, the hidden-state probe
is not evidence that the model "knows" more than surface features carry. It does (AUC 0.94 against 0.87),
and inside the same frontier it solves 22 of the 30 hard problems against 15. No language model needed.
"""
import random

import numpy as np
import torch

from interference_search.countdown import children, gen_hard_problem, gen_problem, reachable_states, solver
from interference_search.judge import PAD, num_feats


def labelled(problems, cap=None, seed=0):
    rows = []
    for p in problems:
        solv = solver(p["target"])
        rows += [(s, p["target"], solv(s)) for s in sorted(reachable_states(p["numbers"]))]
    if cap and len(rows) > cap:
        rows = random.Random(seed).sample(rows, cap)
    return rows


def features(rows):
    X = []
    for s, t, _ in rows:
        per = [num_feats(x, t) for x in sorted(s)] + [PAD] * (7 - len(s))
        glob = [len(s), np.log1p(sum(s)) - np.log1p(t), np.log1p(max(s)) - np.log1p(t), float(t in s), float(sum(s) >= t)]
        X.append(np.concatenate([np.array(per).ravel(), glob]))
    return np.array(X)


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_probe(X, y, Xv, yv):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, Vt = (torch.tensor((a - mu) / sd, dtype=torch.float32) for a in (X, Xv))
    yt = torch.tensor(y, dtype=torch.float32)
    best = (-1, None)
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        w, b = torch.zeros(X.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=wd)
        pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))
        for _ in range(300):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = auc((Vt @ w + b).numpy(), yv)
        if a > best[0]:
            best = (a, (w.detach().clone(), b.detach().clone()))
    w, b = best[1]
    return lambda Z: torch.sigmoid(torch.tensor((Z - mu) / sd, dtype=torch.float32) @ w + b).numpy()


# the same splits as probe.py
rng = random.Random(7)
train_p = [gen_hard_problem(rng) for _ in range(70)]
val_p, train_p = train_p[:10], train_p[10:]
train = labelled(train_p)
alive, dead = [r for r in train if r[2]], [r for r in train if not r[2]]
train = alive + random.Random(0).sample(dead, min(len(dead), 3 * len(alive)))
val = labelled(val_p)
test4 = labelled([gen_hard_problem(random.Random(900 + i)) for i in range(30)])
test5 = labelled([gen_problem(random.Random(1900 + i), n_numbers=5) for i in range(8)], cap=2000)
probe = train_probe(features(train), np.array([r[2] for r in train], float), features(val), np.array([r[2] for r in val], float))
print(f"number-feature probe, {len(train)} training states: AUC 4-number {auc(probe(features(test4)), [r[2] for r in test4]):.3f}"
      f"   5-number {auc(probe(features(test5)), [r[2] for r in test5]):.3f}")

# the same frontier as probe.py, on the same 30 hard problems as llm_state.py
rng11 = random.Random(11)
problems = [gen_hard_problem(rng11) for _ in range(30)]


def enum_frontier(p, width=6, max_evals=150):
    target, live, evals = p["target"], [tuple(sorted(p["numbers"]))], 0
    while live and evals < max_evals:
        kids = sorted({c for s in live for c in children(s)})
        if (target,) in kids:
            return True
        kids = [c for c in kids if len(c) > 1][: max_evals - evals]
        if not kids:
            return False
        sc = probe(features([(k, target, None) for k in kids]))
        evals += len(kids)
        live = [kids[j] for j in np.argsort(-sc)[:width]]
    return False


print("number-feature probe as the judge, 30 hard problems: solved", sum(enum_frontier(p) for p in problems), "/ 30")
