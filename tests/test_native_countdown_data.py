from experiments.native_countdown_data import CODEC, make_row
from interference_search.countdown import solver


def test_curriculum_refutation_crosses_branches_and_labels_are_exact():
    row = make_row(seed=41, n_numbers=4)
    assert row is not None
    refuted = tuple(row["verified_dead_state"])
    can_solve = solver(row["target"])
    assert not can_solve(refuted)
    assert sum(bool(actions) for actions in row["refutation_actions"]) >= 2
    for state_values, alive, action_id, refuted_actions in zip(
        row["frontier"], row["survival_targets"], row["action_targets"], row["refutation_actions"]
    ):
        state = tuple(state_values)
        assert alive == int(can_solve(state))
        for candidate in refuted_actions:
            assert CODEC.legal(state, candidate)
            assert CODEC.apply(state, candidate) == refuted
        if action_id >= 0:
            assert CODEC.legal(state, action_id)
            assert can_solve(CODEC.apply(state, action_id))
        else:
            assert not alive
