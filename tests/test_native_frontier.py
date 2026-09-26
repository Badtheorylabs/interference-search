import torch

from interference_search.native_frontier import NativeFrontierCell


def inputs():
    torch.manual_seed(7)
    states = torch.randn(2, 3, 16)
    task = torch.randn(2, 16)
    refs = torch.randn(2, 2, 16)
    live = torch.tensor([[True, True, False], [True, True, True]])
    verified = torch.tensor([[True, False], [True, True]])
    return states, live, task, refs, verified


def test_verified_refutations_only_lower_survival_scores():
    cell = NativeFrontierCell(hidden=16, actions=4, scale=0.2).eval()
    states, live, task, refs, verified = inputs()
    off = cell(states, live, task, refs, verified, coupling="zero")
    down = cell(states, live, task, refs, verified, coupling="inhibit")
    up = cell(states, live, task, refs, verified, coupling="excite")
    torch.testing.assert_close(down.survival_logits[live],
                               (off.survival_logits - 0.2 * down.refutation_match)[live])
    torch.testing.assert_close(up.survival_logits[live],
                               (off.survival_logits + 0.2 * up.refutation_match)[live])
    torch.testing.assert_close(down.proposal_logits[live],
                               (off.proposal_logits - 0.2 * down.action_refutation_match)[live])
    torch.testing.assert_close(up.proposal_logits[live],
                               (off.proposal_logits + 0.2 * up.action_refutation_match)[live])
    assert (down.refutation_match[live] >= 0).all()
    assert (down.survival_logits[live] <= off.survival_logits[live]).all()
    empty = cell(states, live, task, refs[:, :0], verified[:, :0], coupling="inhibit")
    torch.testing.assert_close(empty.survival_logits[live], off.survival_logits[live])


def test_hard_exclusion_is_separate_from_soft_similarity():
    cell = NativeFrontierCell(hidden=16, actions=4).eval()
    states, live, task, refs, verified = inputs()
    hard = torch.tensor([[False, True, False], [False, False, True]])
    out = cell(states, live, task, refs, verified, hard_exclusion=hard)
    assert (out.survival_logits[hard] == -1e9).all()
    assert (out.proposal_logits[hard] == -1e9).all()
    assert (out.survival_logits[live & ~hard] > -1e9).all()
    hard_actions = torch.zeros(2, 3, 4, dtype=torch.bool)
    hard_actions[0, 0, 2] = True
    action_out = cell(states, live, task, refs, verified, hard_action_exclusion=hard_actions)
    assert action_out.proposal_logits[0, 0, 2] == -1e9
    assert action_out.proposal_logits[0, 0, 1] > -1e9


def test_refutation_reaches_same_proposed_action_across_states():
    cell = NativeFrontierCell(hidden=16, actions=3).eval()
    states, live, task, refs, verified = inputs()
    states[0, 1] = states[0, 0]
    proposed = torch.randn(2, 3, 3, 16)
    proposed[0, 1, 0] = proposed[0, 0, 0]
    out = cell(states, live, task, refs, verified, action_vectors=proposed)
    torch.testing.assert_close(out.predicted_successors[0, 0, 0],
                               out.predicted_successors[0, 1, 0])
    torch.testing.assert_close(out.action_refutation_match[0, 0, 0],
                               out.action_refutation_match[0, 1, 0])


def test_frontier_is_permutation_equivariant_and_trains():
    cell = NativeFrontierCell(hidden=16, actions=4)
    states, live, task, refs, verified = inputs()
    before = cell(states, live, task, refs, verified)
    order = torch.tensor([2, 0, 1])
    shuffled = cell(states[:, order], live[:, order], task, refs, verified)
    torch.testing.assert_close(before.survival_logits[:, order], shuffled.survival_logits,
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before.proposal_logits[:, order], shuffled.proposal_logits,
                               atol=1e-6, rtol=1e-5)

    old = cell.query.weight.detach().clone()
    loss = before.survival_logits[live].square().mean() + before.proposal_logits[live].square().mean()
    loss.backward()
    assert cell.query.weight.grad is not None
    assert cell.query.weight.grad.abs().sum() > 0
    torch.optim.AdamW(cell.parameters(), lr=1e-3).step()
    assert not torch.equal(old, cell.query.weight.detach())
