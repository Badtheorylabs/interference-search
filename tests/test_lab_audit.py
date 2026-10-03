"""Tests for the lab audit and diagnostic tools. Each pins a finding recorded
in docs/NATIVE_LATENT_INTERFERENCE_ROADMAP.md, so a fix that changes the
finding must also update the ledger."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "lab"))

import audit_data
import envs


def test_split_holds_out_last_two_per_group():
    g = envs.make_dataset(seed=0)
    train, test = audit_data.split(g)
    expected = sum(min(len(x["seqs"]), 8) - max(1, min(len(x["seqs"]), 8) - 2) for x in g)
    assert len(test) == expected
    assert not {r[0] for r in train} & {r[0] for r in test}


def test_audit_findings_pinned(capsys):
    audit_data.main()
    out = capsys.readouterr().out
    assert "192 groups from 2 start sets" in out
    assert "negatives from same start set (hard negatives): 0/" in out
    assert "shortcut equivalence AUC (shared operand count, no state tracking): 0.99" in out
    assert "depth 3: 206 test rows | state = final move result only: 206" in out


try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


@needs_torch
def test_slot_stats_detects_collapse():
    from diagnose import slot_stats
    same = torch.ones(2, 4, 8)
    assert slot_stats(same)["slot_cos"] == pytest.approx(1.0)
    assert slot_stats(same)["rel_spread"] == pytest.approx(0.0)
    distinct = torch.eye(8)[:4].unsqueeze(0).repeat(2, 1, 1)
    assert slot_stats(distinct)["slot_cos"] == pytest.approx(0.0)


@needs_torch
def test_merge_isolated_reports_reflection_at_init():
    from diagnose import build_model, merge_isolated
    torch.manual_seed(0)
    r = merge_isolated(build_model("interference"))
    assert r["sum_j_equiv_x_alpha"] > 1.0              # coefficients overshoot
    assert r["cos(deviation_before, deviation_after)"] < 0  # reflection, not contraction


@needs_torch
def test_trace_and_dead_params():
    from diagnose import MODES, build_model, loss_of, trace
    torch.manual_seed(0)
    m = build_model("interference")
    ids = torch.tensor([envs.encode_seq(["004+026", "030*028"])] * 3, device=next(m.parameters()).device)
    rows = trace(m.eval(), ids, ids != 0, MODES["interference"])
    assert len(rows) == int((ids != 0).sum(1).max()) + 1
    assert all(torch.isfinite(torch.tensor(r["norm"])) for r in rows)
    tgt = torch.tensor([envs.encode_state((56, 840))] * 3, device=ids.device)
    m.train().zero_grad()
    loss_of(m, "interference", ids, tgt).backward()
    dead = {n for n, p in m.named_parameters() if p.grad is None}
    assert {"unit.up.weight", "head.weight", "pair_head.weight"} <= dead


@needs_torch
def test_frontier_blows_up_at_init_for_seed_1():
    from diagnose import MODES, build_model, trace
    from run_lab import build_rows
    torch.manual_seed(1)
    (ids, _), _ = build_rows(envs.make_dataset(seed=1))
    torch.manual_seed(1)
    m = build_model("interference").eval()
    ids = ids[:64].to(next(m.parameters()).device)
    norms = [r["norm"] for r in trace(m, ids, ids != 0, MODES["interference"])]
    assert max(norms) > 1e4 * norms[1]                 # unnormalized recurrence at init


@needs_torch
def test_compute_measure_runs(capsys):
    import measure_compute
    measure_compute.main()
    out = capsys.readouterr().out
    ratio = float(out.split("FLOPS_RATIO interference/vanilla (fwd+bwd, train mode): ")[1].split()[0])
    assert 0.8 < ratio < 1.2                            # equal steps is equal compute


@needs_torch
def test_collapse_probe_runs(capsys):
    import collapse_probe
    collapse_probe.main()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "merge=off" in l and "prop=learned" in l)
    assert float(line.split("slot cosine after 8 steps ")[1].split(",")[0]) > 0.99
