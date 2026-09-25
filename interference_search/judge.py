"""The Countdown judge: a small set transformer that scores how likely a state can still reach the target.

A state is encoded as one feature vector per number, each describing the number relative to the
target (log size, distance, divisibility, residues). The numbers attend to each other, a class token
pools them, and one logit comes out: alive or dead. The model is permutation-invariant and works for
any count of numbers, which is what lets a judge trained on small problems score bigger ones.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn

MODS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "countdown_judge.pt"


def num_feats(x, t):
    lx, lt = math.log1p(x), math.log1p(t)
    f = [lx / 10, (lx - lt) / 5, float(x == t), float(x > t), float(t % x == 0), float(x % t == 0),
         min(abs(x - t), 1000) / 1000]
    f += [float(x % m == t % m) for m in MODS]
    f += [((x % m) / m) for m in MODS]
    return f


N_FEAT = len(num_feats(3, 7))
PAD = [0.0] * N_FEAT


def encode(states, target, max_n=7):
    """Batch of states -> (features, mask). `target` is one int or a list with one per state."""
    ts = target if isinstance(target, list) else [target] * len(states)
    feats, mask = [], []
    for s, t in zip(states, ts):
        feats.append([num_feats(x, t) for x in s] + [PAD] * (max_n - len(s)))
        mask.append([True] * len(s) + [False] * (max_n - len(s)))
    return torch.tensor(feats), torch.tensor(mask)


class Refuter(nn.Module):
    """Set transformer: numbers attend to each other, then pooled to one alive/dead logit."""

    def __init__(self, d=64, layers=2):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, X, M):
        h = torch.cat([self.cls.expand(X.shape[0], -1, -1), self.inp(X)], 1)
        mask = torch.cat([torch.ones(X.shape[0], 1, dtype=torch.bool), M], 1)
        h = self.tf(h, src_key_padding_mask=~mask)
        return self.out(h[:, 0]).squeeze(-1)          # logit of ALIVE


def load_judge(path=DEFAULT_WEIGHTS):
    """Load trained weights and return score(states, target) -> list of P(alive)."""
    model = Refuter()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    def score(states, target):
        if not states:
            return []
        with torch.no_grad():
            X, M = encode(list(states), target)
            return torch.sigmoid(model(X, M)).tolist()
    score.model = model
    return score


def upper_bound(state):
    """A sound upper bound on any value reachable from a state (a 1 can only add)."""
    return math.prod(max(x, 2) for x in state) if len(state) > 1 else state[0]


def hand_alive(state, target):
    """The classical baseline: prune only when even the upper bound cannot reach the target."""
    return upper_bound(state) >= target
