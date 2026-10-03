"""Exact-search Countdown environment for native frontier computation.

Move rules come from the paper's solver (`interference_search.countdown.children`),
restricted to values <= CAP so every value fits one decoder. Start sets are 4
numbers from 1..N_MAX. Task: given the start numbers and a target, can the
target be reached using every number once? Every label the lab uses (move
results, state identity, which states can still reach the target) is computed
exactly by enumeration. No judge anywhere.

Train and test use disjoint start sets, and labels are balanced per target, so
neither "seen this start before" nor "this target is usually reachable" helps.
"""
import itertools
import os
import random
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from interference_search.countdown import children as paper_children  # noqa: E402

N_MAX, CAP, N_NUMS = 13, 120, 4
V = CAP + 1                 # value classes; 0 means "no number" / illegal move
OPS = ("+", "*", "-", "/")


@lru_cache(maxsize=None)
def kids(state):
    """Distinct states one move away (the paper's merging environment), values <= CAP."""
    return tuple(c for c in paper_children(state) if max(c) <= CAP)


@lru_cache(maxsize=None)
def finals(state):
    """Every value reachable as the last remaining number."""
    if len(state) == 1:
        return frozenset(state)
    return frozenset().union(*(finals(c) for c in kids(state))) if kids(state) else frozenset()


def apply(x, y, op):
    """The model's move semantics on an unordered pair: - and / act on
    (larger, smaller). Returns 0 when the move is illegal. Over all pairs and
    ops this reproduces `kids` exactly (tested)."""
    a, b = max(x, y), min(x, y)
    r = {"+": a + b, "*": a * b, "-": a - b, "/": a // b if b > 1 and a % b == 0 else 0}[op]
    return r if 0 < r <= CAP else 0


def all_starts():
    return list(itertools.combinations_with_replacement(range(1, N_MAX + 1), N_NUMS))


def state_key(state):
    """Multiset -> integer. Zero padded to N_NUMS, sorted, base V."""
    s = sorted(list(state) + [0] * (N_NUMS - len(state)))
    return sum(v * V ** k for k, v in enumerate(s))


def all_states():
    """Every state reachable from any start set, including final single numbers."""
    seen, stack = set(), list(all_starts())
    while stack:
        s = stack.pop()
        if s not in seen:
            seen.add(s)
            stack.extend(kids(s))
    return sorted(seen)


def split_starts(seed, test_frac=0.2):
    starts = all_starts()
    random.Random(seed).shuffle(starts)
    n = int(len(starts) * test_frac)
    return sorted(starts[n:]), sorted(starts[:n])


def by_target(starts):
    """target -> (reachable start sets, unreachable start sets)."""
    out = {}
    for t in range(1, V):
        pos = [s for s in starts if t in finals(s)]
        neg = [s for s in starts if t not in finals(s)]
        if pos and neg:
            out[t] = (pos, neg)
    return out


def sample(index, n, rng):
    """(start, target, label) with the target drawn first and labels balanced
    per target, so the target alone carries no information about the label."""
    targets = sorted(index)
    out = []
    for _ in range(n):
        t = rng.choice(targets)
        y = rng.random() < 0.5
        out.append((rng.choice(index[t][0 if y else 1]), t, int(y)))
    return out


def eval_set(starts, seed, per_target=6):
    """Exactly per_target positives and negatives for every usable target."""
    rng = random.Random(seed + 7)
    out = []
    for t, (pos, neg) in sorted(by_target(starts).items()):
        m = min(per_target, len(pos), len(neg))
        out += [(s, t, 1) for s in rng.sample(pos, m)] + [(s, t, 0) for s in rng.sample(neg, m)]
    return out


def auc(scores, labels):
    """Probability a positive outranks a negative, ties counted half."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def shortcut_scores(train_starts, examples, k=5):
    """Ways to score reachability without searching. A test that a shortcut
    passes cannot show search.

    target_prior: how often the target is reachable across training start sets.
    nearest_start: reachability of the target from the k nearest training start sets.
    """
    prior = {t: sum(t in finals(s) for s in train_starts) / len(train_starts) for t in range(1, V)}
    out = {"target_prior": [], "nearest_start": []}
    for st, t, _ in examples:
        near = sorted(train_starts, key=lambda s: sum(abs(a - b) for a, b in zip(s, st)))[:k]
        out["target_prior"].append(prior[t])
        out["nearest_start"].append(sum(t in finals(s) for s in near) / k)
    labels = [y for _, _, y in examples]
    return {name: auc(v, labels) for name, v in out.items()}


def exact_search(start, target, width, judge, merge, rng):
    """The paper's algorithm with exact moves, at a fixed frontier width.

    judge=True keeps states that can still reach the target first (an oracle
    judge); judge=False keeps a random subset. merge=False keeps one copy per
    path instead of per state. Returns True when the target is found. Used as
    a reference for how tight `width` is, not as a model.
    """
    frontier = [start]
    for _ in range(len(start) - 1):
        cands = [c for s in frontier for c in (kids(s) if merge else
                 [tuple(sorted([x for k, x in enumerate(s) if k not in (i, j)] + [r]))
                  for i, j in itertools.combinations(range(len(s)), 2)
                  for r in (apply(s[i], s[j], op) for op in OPS) if r])]
        if merge:
            cands = sorted(set(cands))
        rng.shuffle(cands)
        if judge:
            cands.sort(key=lambda c: target not in finals(c))
        frontier = cands[:width]
    return any(s == (target,) for s in frontier)
