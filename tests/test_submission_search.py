import random

from interference_search.countdown import moves, solver
from experiments.submission.build_countdown import path_count, stratum
from experiments.submission.countdown_engine import Method, methods, run


def test_every_arm_obeys_hard_work_cap_and_has_legal_witness():
    problem = {"numbers": [2, 3, 7, 10], "target": 25}
    def score(states, target):
        exact = solver(target)
        return [float(exact(s)) for s in states]
    for arm in methods():
        for budget in (0, 1, 3, 10, 50, 200):
            result = run(problem, arm, 7, budget, score)
            assert result["work_used"] <= budget
            assert result["work_used"] == sum(result[k] for k in
                   ("states_proposed", "states_expanded", "states_judged"))
            if result["solved"]:
                path = result["solution_state_path"]
                assert path[0] == tuple(sorted(problem["numbers"]))
                assert path[-1] == (problem["target"],)
                assert all(b in moves(a) for a,b in zip(path,path[1:]))


def test_merging_is_not_terminal_failure_or_pruning():
    p = {"numbers": [2, 2, 3], "target": 997}
    off = run(p, Method("off", False, False, True), 0, 1000, lambda s,t: [])
    on = run(p, Method("on", True, False, True), 0, 1000, lambda s,t: [])
    assert off["states_merged"] == 0
    assert off["terminal_failures"] > 0
    assert on["states_merged"] > 0
    assert on["states_pruned"] == 0
    assert all(r["unique_states"] <= r["raw_candidates"] for r in on["depth_trace"])


def test_synchrony_reduces_dependency_rounds_without_hiding_expansions():
    p = {"numbers": [2, 3, 7, 10], "target": 997}
    sync = run(p, Method("sync", True, False, True), 2, 10000, lambda s,t: [], width=10000)
    bfs = run(p, Method("bfs", True, False, False, "bfs"), 2, 10000, lambda s,t: [])
    assert sync["sequential_rounds"] == 3
    assert bfs["sequential_rounds"] == bfs["states_expanded"]
    assert sync["states_expanded"] == bfs["states_expanded"]


def test_seed_repeat_and_three_switch_factorial():
    p = {"numbers": [2, 3, 7, 10], "target": 25}
    arm = methods()[0]
    a = run(p, arm, 4, 200, lambda s,t: [])
    b = run(p, arm, 4, 200, lambda s,t: [])
    a.pop("wall_seconds")
    b.pop("wall_seconds")
    assert a == b
    cells = [m for m in methods() if m.name.startswith("factorial_")]
    assert len({(m.merge,m.judge,m.synchronous) for m in cells}) == 8


def test_exact_path_counts_keep_duplicate_operation_sequences():
    p = {"numbers": [2,2], "target": 4}
    assert path_count(p) == 2  # addition and multiplication
    assert [stratum(n) for n in (1,2,5,6,20,21)] == ["1","2-5","2-5","6-20","6-20",">20"]


def test_judge_encoding_handles_eight_number_states_in_mixed_batch():
    from interference_search.judge import encode
    features,mask=encode([(1,2,3,4,5,6,7,8),(2,4)],19)
    assert features.shape[:2] == (2,8)
    assert mask.sum(1).tolist() == [8,2]
