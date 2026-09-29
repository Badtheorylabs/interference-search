"""Exact-state Countdown environment for the interference lab.

Deterministic enumeration. From one number set, every legal pair-operation is
a move; a state is the sorted multiset of numbers remaining. A trajectory is a
sequence of move tokens. Two trajectories arriving at the same multiset are
equivalent states by definition: identity is exact arithmetic, computed by the
enumeration, with no judge involved.

Encoding: each move token "abc+def" is encoded character-by-character over a
16-symbol alphabet, moves separated by the "|" token. The model reads the move
characters and must predict the remaining numbers (exact multiset) at the
trajectory's end. Outcome-only supervision: the target is the true state.
"""
import itertools
import random

ALPHABET = "0123456789+-*/|"
PAD = 0
TOK_MAP = {c: i + 1 for i, c in enumerate(ALPHABET)}
SEQ_PAD = 24


def encode_seq(moves):
    """Encode a move sequence into token ids, padded to SEQ_PAD."""
    s = "|".join(moves)
    ids = [TOK_MAP[c] for c in s]
    return ids[:SEQ_PAD] + [PAD] * (SEQ_PAD - len(ids[:SEQ_PAD]))


def encode_state(state, width=3):
    """Sorted remaining numbers, each zero-padded to 3 digits, fixed total width
    numbers (pad slots "000"), as digit CLASS ids 0..9. Exact: 000 is a valid
    padding marker because real remaining numbers are never 000."""
    nums = sorted(int(v) for v in state)
    while len(nums) < width:
        nums.append(0)
    return [int(d) for n in nums[:width] for d in f"{n:03d}"]


def state_vocab_len():
    return 10          # digits only, padded fixed width


def all_moves(state):
    out = []
    for i, j in itertools.permutations(range(len(state)), 2):
        a, b = state[i], state[j]
        cands = [("+", a + b), ("-", a - b), ("*", a * b),
                 ("/", a / b if b and a % b == 0 else None)]
        for op, r in cands:
            if r is None or r <= 0 or int(r) != r or r > 999:
                continue
            rest = tuple(sorted([x for k, x in enumerate(state) if k not in (i, j)] + [int(r)]))
            out.append((f"{a:03d}{op}{b:03d}", rest))
    return out


def enumerate_trajectories(nums, depth=None):
    """Full move sequences from the start state to each depth, keyed by state.
    levels[d][state] = list of distinct move-token sequences reaching state."""
    start = tuple(sorted(nums))
    levels = [{start: [()]}]
    frontier = {start}
    max_depth = len(nums) - 1 if depth is None else depth
    for _ in range(max_depth):
        nxt = {}
        for s in frontier:
            for tok, s2 in all_moves(s):
                for prefix in levels[-1][s]:
                    nxt.setdefault(s2, []).append(prefix + (tok,))
        frontier = set(nxt)
        levels.append(nxt)
    return levels


def make_dataset(seed=0, n_groups=48, min_diversity=4):
    """Groups of distinct trajectories reaching the same exact depth-2 multiset.
    Each group is one exact state with genuine trajectory diversity."""
    rng = random.Random(seed)
    groups, done_nums = [], []
    attempts = 0
    while len(groups) < n_groups and attempts < 4000:
        attempts += 1
        nums = sorted([rng.randint(2, 40) for _ in range(3)] + [rng.randint(40, 99)])
        if nums in done_nums:
            continue
        done_nums.append(nums)
        levels = enumerate_trajectories(nums)
        if len(levels) < 3:
            continue
        best = max(levels[2].values(), key=len, default=[])
        if len(best) < min_diversity:
            continue
        state = max(levels[2], key=lambda s: len(levels[2][s]))
        seqs = levels[2][state][:8]
        groups.append({"nums": nums, "state": state, "seqs": seqs})
    return groups


def encode_example(group):
    """All rows for a group: each trajectory -> exact remaining state."""
    rows = []
    for seq in group["seqs"]:
        rows.append((encode_seq(list(seq)), encode_state(group["state"])))
    return rows


def make_pairs(groups, n_pairs=400, seed=1):
    """Trajectory pairs with exact identity labels: same multiset (same group)
    vs different multiset."""
    rng = random.Random(seed)
    pairs, labels = [], []
    same_ok = [g for g in groups if len(g["seqs"]) >= 2]
    while len(pairs) < n_pairs:
        if same_ok and rng.random() < 0.5:
            g = rng.choice(same_ok)
            a, b = rng.sample(g["seqs"], 2)
            pairs.append((list(a), list(b))); labels.append(1)
        else:
            g1, g2 = rng.choice(groups), rng.choice(groups)
            a = rng.choice(g1["seqs"]); b = rng.choice(g2["seqs"])
            if g1["state"] == g2["state"] and g1["nums"] == g2["nums"]:
                continue
            pairs.append((list(a), list(b))); labels.append(0)
    return pairs, labels
