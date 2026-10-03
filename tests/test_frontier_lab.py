"""Tests for the native frontier lab: exact environment, move semantics, and
the frontier network's invariants."""
import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "lab"))

import frontier_env as E


def test_kids_match_paper_children_under_cap():
    from interference_search.countdown import children
    for s in E.all_starts()[::50]:
        assert set(E.kids(s)) == {c for c in children(s) if max(c) <= E.CAP}


def test_apply_reproduces_environment_for_every_state():
    for s in E.all_states():
        if len(s) < 2:
            continue
        mine = {tuple(sorted([x for k, x in enumerate(s) if k not in (i, j)] + [r]))
                for i, j in itertools.combinations(range(len(s)), 2)
                for r in (E.apply(s[i], s[j], op) for op in E.OPS) if r}
        assert mine == set(E.kids(s))


def test_finals_is_exact_reachability():
    assert E.finals((2, 3)) == {1, 5, 6}
    assert 24 in E.finals((1, 2, 3, 4))


def test_train_test_starts_disjoint_and_cover():
    tr, te = E.split_starts(0)
    assert not set(tr) & set(te)
    assert len(tr) + len(te) == len(E.all_starts())


def test_eval_set_balanced_per_target():
    _, te = E.split_starts(0)
    ev = E.eval_set(te, 0)
    for t in {t for _, t, _ in ev}:
        ys = [y for _, tt, y in ev if tt == t]
        assert sum(ys) * 2 == len(ys)
        for s, tt, y in ev:
            if tt == t:
                assert (t in E.finals(s)) == bool(y)


def test_target_alone_carries_no_signal():
    tr, te = E.split_starts(0)
    assert E.shortcut_scores(tr, E.eval_set(te, 0))["target_prior"] == pytest.approx(0.5)


def test_auc_basic():
    assert E.auc([0.1, 0.9], [0, 1]) == 1.0
    assert E.auc([0.5, 0.5], [0, 1]) == 0.5
    assert E.auc([1, 1], [1, 1]) is None


def test_exact_search_oracle_judge_always_solves():
    rng = random.Random(0)
    _, te = E.split_starts(0)
    pos = [(s, t) for s, t, y in E.eval_set(te, 0) if y][:100]
    assert all(E.exact_search(s, t, 1, True, True, rng) for s, t in pos)


try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


@needs_torch
def test_exact_apply_matches_environment():
    from frontier_model import OPS, exact_apply
    xs = torch.arange(0, 20).repeat_interleave(20)
    ys = torch.arange(0, 20).repeat(20)
    for k, op in enumerate(OPS):
        got = exact_apply(xs, ys, torch.full_like(xs, k)).tolist()
        want = [E.apply(x, y, op) if x and y else 0 for x, y in zip(xs.tolist(), ys.tolist())]
        assert got == want


@needs_torch
@pytest.mark.parametrize("cfg", [dict(survival=False, interference=False, merge=False),
                                 dict(survival=True, interference=False, merge=False),
                                 dict(survival=True, interference=True, merge=True)])
def test_forward_shapes_and_true_values_tracked(cfg):
    from frontier_model import FrontierNet
    torch.manual_seed(0)
    m = FrontierNet(d=32, k=3, **cfg).eval()
    starts = torch.tensor([[2, 3, 4, 5], [1, 1, 7, 13]])
    targets = torch.tensor([14, 100])
    p, hit_mass, aux, rec = m(starts, targets, record=True)
    assert p.shape == (2,) and hit_mass.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()
    assert len(aux) == E.N_NUMS - 1
    # every kept state is either a real reachable state or marked dead (all
    # zeros); never a partial state with a number missing
    for step in (0, 1):
        st = rec[f"step{step}"]
        for b in range(2):
            level = {tuple(starts[b].tolist())}
            for _ in range(step + 1):
                level = {c for s in level for c in E.kids(s)}
            for j, ok in zip(st["kept"][b].tolist(), st["kept_ok"][b].tolist()):
                vals = st["ev"][b, j].tolist()
                if ok and any(vals):
                    assert tuple(sorted(v for v in vals if v)) in level


@needs_torch
def test_illegal_move_kills_true_state():
    from frontier_model import FrontierNet
    m = FrontierNet(d=16, k=2).eval()
    starts = torch.tensor([[2, 3, 5, 7]])
    n = m.norm(m.num(starts)).unsqueeze(1)
    _, _, _, _, _, r, cev = m.expand(n, (starts > 0).unsqueeze(1), starts.unsqueeze(1))
    dead = r == 0
    assert dead.any()
    assert (cev[dead] == 0).all()
    assert (cev[~dead] > 0).sum(-1).eq(3).all()


@needs_torch
def test_outcome_loss_cannot_train_hit_decoding():
    """The final hit is scored with detached decoder and legality heads, so the
    outcome loss cannot make the network claim a hit the arithmetic did not
    produce. The arithmetic layer does receive outcome gradient through the
    survival scores of states built from earlier moves; that shapes the state
    space but cannot fake a hit, and `frontier_solve_rate` is checked on true
    values."""
    from frontier_model import FrontierNet
    torch.manual_seed(0)
    m = FrontierNet(d=32, k=3).train()
    _, hit_mass, _, _ = m(torch.tensor([[2, 3, 4, 5]]), torch.tensor([14]))
    (-torch.log(hit_mass + 1e-6)).sum().backward()
    for name in ("legal", "dec_bias", "log_scale"):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(name)]
        assert all(g is None or g.abs().sum() == 0 for g in grads), name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("score"))


@needs_torch
def test_merge_only_collapses_near_identical_states():
    from frontier_model import FrontierNet
    m = FrontierNet(d=8, k=2, survival=False, interference=False, merge=True).eval()
    c = torch.tensor([[[1.0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.01, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    gen = torch.Generator().manual_seed(0)
    idx, mass, ok, _ = m.select(c, c, torch.zeros(1, 8), valid, gen)
    kept = set(idx[0][ok[0]].tolist())
    assert 2 in kept and len(kept) == 2        # the two near-copies became one
