"""Data audit for the interference lab. Pure Python, no torch.

Answers: what does the split actually test, how hard are the equivalence
pairs, and can a shortcut solve the task without tracking state?
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs


def split(groups):
    train, test = [], []
    for g in groups:
        seqs = g["seqs"][:8]
        n_keep = max(1, len(seqs) - 2)
        for i, s in enumerate(seqs):
            (train if i < n_keep else test).append((tuple(s), g["state"], g["nums"]))
    return train, test


def main():
    for seed in (0, 991):
        g = envs.make_dataset(seed=seed)
        ns = Counter(x["nums"] for x in g)
        depths = Counter(len(x["seqs"][0]) for x in g)
        sizes = sorted(len(x["seqs"]) for x in g)
        print(f"seed {seed}: {len(g)} groups from {len(ns)} start sets")
        print(f"  groups per start set: {sorted(ns.values(), reverse=True)[:12]} ...")
        print(f"  trajectory depth: {dict(depths)}")
        print(f"  seqs/group min/median/max: {sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")

    g = envs.make_dataset(seed=0)
    train, test = split(g)
    print(f"\ntrain rows {len(train)}, test rows {len(test)}")
    enc_lens = Counter(sum(1 for t in envs.encode_seq(list(s)) if t) for s, _, _ in train + test)
    print(f"encoded token lengths: {dict(sorted(enc_lens.items()))}  (SEQ_PAD={envs.SEQ_PAD})")
    raw_lens = Counter(len("|".join(s)) for s, _, _ in train + test)
    print(f"raw string lengths before truncation: {dict(sorted(raw_lens.items()))}")

    # Shortcut 1: the final move token alone
    last2state = {}
    for s, st, _ in train:
        last2state.setdefault(s[-1], Counter())[st] += 1
    hit = sum(1 for s, st, _ in test if s[-1] in last2state
              and last2state[s[-1]].most_common(1)[0][0] == st)
    print(f"shortcut 'final move -> most common train state': {hit}/{len(test)} test rows correct")

    # Shortcut 2: memorize state per start set (start set = first move numbers)
    test_states_seen = sum(1 for _, st, _ in test if st in {x[1] for x in train})
    print(f"test rows whose target state appears in train: {test_states_seen}/{len(test)}")

    # Equivalence pairs
    eg = envs.make_dataset(seed=991)
    pairs, labels = envs.make_pairs(eg)
    seq2nums = {tuple(s): x["nums"] for x in eg for s in x["seqs"]}
    pos = [(a, b) for (a, b), y in zip(pairs, labels) if y == 1]
    neg = [(a, b) for (a, b), y in zip(pairs, labels) if y == 0]
    print(f"\nequiv pairs: {len(pos)} pos, {len(neg)} neg")
    print(f"  positives from same start set: {sum(seq2nums[tuple(a)] == seq2nums[tuple(b)] for a, b in pos)}/{len(pos)}")
    print(f"  negatives from same start set (hard negatives): "
          f"{sum(seq2nums[tuple(a)] == seq2nums[tuple(b)] for a, b in neg)}/{len(neg)}")
    train_states = {x["state"] for x in g}
    eval_states = {x["state"] for x in eg}
    print(f"  eval states also present in seed-0 training: {len(train_states.intersection(eval_states))}/{len(eval_states)}")
    same_depth = sum(len(a) == len(b) for a, b in neg)
    print(f"  negatives with equal trajectory depth: {same_depth}/{len(neg)}")

    print("\nexample group:", g[0]["nums"], "->", g[0]["state"])
    for s in g[0]["seqs"][:4]:
        print("   ", s)

    # Near-duplicate leakage: a*b vs b*a, a+b vs b+a are different token strings
    # for the same computation.
    def canon(move):
        a, op, b = move[:3], move[3], move[4:]
        return (min(a, b), op, max(a, b)) if op in "+*" else (a, op, b)

    train_canon = {tuple(canon(m) for m in s) for s, _, _ in train}
    twin = sum(tuple(canon(m) for m in s) in train_canon for s, _, _ in test)
    print(f"\ntest rows whose commutative twin is in train: {twin}/{len(test)}")
    first_move = {}
    for s, st, _ in train:
        first_move.setdefault(st, set()).add(s[0])
    shared_first = sum(s[0] in first_move.get(st, set()) for s, st, _ in test)
    print(f"test rows sharing their first move with a train row of the same state: {shared_first}/{len(test)}")
    distinct_canon = Counter(len({tuple(canon(m) for m in s) for s in x["seqs"][:8]}) for x in g)
    print(f"distinct computations among the 8 used seqs per group: {dict(sorted(distinct_canon.items()))}")

    # Shortcut 3: equivalence AUC from a rule that never tracks state. Score a
    # pair by how many 3-digit operands its two trajectories share.
    def operands(seq):
        return {m[:3] for m in seq} | {m[4:] for m in seq}

    scores = [len(operands(a) & operands(b)) for a, b in pairs]
    ps = [s for s, y in zip(scores, labels) if y == 1]
    ns_ = [s for s, y in zip(scores, labels) if y == 0]
    auc = (sum(p > n for p in ps for n in ns_) + 0.5 * sum(p == n for p in ps for n in ns_)) / (len(ps) * len(ns_))
    print(f"\nshortcut equivalence AUC (shared operand count, no state tracking): {auc:.3f}")

    # Observability: the input is the move string only, never the start set.
    # Replay the moves: start numbers never consumed stay invisible to the model.
    by_depth, last_only, needs_invisible, computable = Counter(), Counter(), Counter(), Counter()
    for s, st, nums in test:
        d = len(s)
        by_depth[d] += 1
        pool = list(nums)
        for m in s:
            a, op, b = int(m[:3]), m[3], int(m[4:])
            pool.remove(a); pool.remove(b)
            pool.append({"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op])
        assert tuple(sorted(pool)) == st
        consumed = list(nums)
        for m in s:
            for v in (int(m[:3]), int(m[4:])):
                if v in consumed:
                    consumed.remove(v)
        if consumed:                      # untouched start numbers remain in the state
            needs_invisible[d] += 1
        elif len(st) == 1:
            last_only[d] += 1
        else:
            computable[d] += 1
    for d in sorted(by_depth):
        print(f"depth {d}: {by_depth[d]} test rows | state = final move result only: {last_only[d]}"
              f" | needs a start number absent from the input: {needs_invisible[d]}"
              f" | multi-number state computable from input: {computable[d]}")


if __name__ == "__main__":
    main()
