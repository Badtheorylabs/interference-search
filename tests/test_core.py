import random

from interference_search import linear_search, search
from interference_search.countdown import solver
from interference_search.countdown_domain import Countdown
from interference_search.judge import load_judge

PROBLEM = {"numbers": [24, 98, 19, 3], "target": 361}


def oracle_judge(states, target):
    solvable = solver(target)
    return [float(solvable(s)) for s in states]


def test_search_with_an_exact_judge_solves_in_three_rounds():
    r = search(Countdown(oracle_judge), PROBLEM, budget=500, width=6)
    assert r.solved and r.rounds == 3 and r.solution == (361,)


def test_merging_removes_duplicates():
    r = search(Countdown(oracle_judge), PROBLEM, budget=500, width=12)
    assert r.merged_away > 0


def test_linear_search_uses_the_same_parts():
    r = linear_search(Countdown(oracle_judge), PROBLEM, budget=500, rng=random.Random(0))
    assert r.solved


def test_trained_judge_loads_and_ranks_alive_states_higher():
    judge = load_judge()
    alive, dead = (3, 98, 456), (2, 3, 5)
    a, d = judge([alive, dead], 361)
    assert 0.0 <= d < a <= 1.0


def test_budget_is_respected():
    r = search(Countdown(oracle_judge), {"numbers": [2, 3, 5, 7], "target": 997}, budget=20, width=4)
    assert r.cost <= 20 + 4 * 30          # at most one round past the budget
