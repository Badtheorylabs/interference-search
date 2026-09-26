import math

from interference_search.countdown import parse_expr, solver
from interference_search.countdown_parallel import CountdownParallelDomain, replay_expression
from interference_search.parallel_reasoning import ParallelReasoner, Transition


class SmallDomain:
    def __init__(self, edges, scores=None, dead=(), goal="G"):
        self.edges = edges
        self.scores = scores or {}
        self.dead = set(dead)
        self.goal = goal

    def start(self, problem):
        return "S"

    def key(self, state):
        return state

    def propose_batch(self, states, problem):
        return [[Transition(child, action=f"{state}->{child}", log_prior=weight)
                 for child, weight in self.edges.get(state, ())] for state in states]

    def judge(self, states, problem):
        return [self.scores.get(state, 0.5) for state in states]

    def is_goal(self, state, problem):
        return state == self.goal

    def certified_dead(self, state, problem):
        return state in self.dead


def test_merged_paths_pool_mass_before_the_next_level():
    half = -math.log(2)
    domain = SmallDomain({"S": [("A", half), ("B", half)],
                          "A": [("C", 0.0)], "B": [("C", 0.0)],
                          "C": [("G", 0.0)]})
    result = ParallelReasoner(width=2, mass_weight=1).run(domain, None, expansion_budget=10)
    assert result.solved
    assert result.merged_proposals >= 1
    assert result.graph["C"].log_mass == 0.0
    assert result.expansions == 4
    assert result.rounds == 3


def test_verified_failure_propagates_and_reserve_recovers_alternative():
    domain = SmallDomain({"S": [("A", 0.0), ("B", 0.0)],
                          "A": [("X", 0.0)], "B": [("G", 0.0)]},
                         scores={"A": 0.9, "B": 0.1}, dead={"X"})
    rescued = ParallelReasoner(width=1, reserve=True).run(domain, None, expansion_budget=3)
    abandoned = ParallelReasoner(width=1, reserve=False).run(domain, None, expansion_budget=3)
    assert rescued.solved and rescued.rescue_rounds == 1
    assert rescued.witness_actions == ["S->B", "B->G"]
    assert not rescued.graph["S"].verified_dead
    assert not abandoned.solved
    assert rescued.expansions == 3 and abandoned.expansions == 2


def test_dead_proof_waits_until_every_child_is_known():
    domain = SmallDomain({"S": [("A", 0.0)], "A": [("X", 0.0), ("Y", 0.0)]},
                         dead={"X"}, goal="not-here")
    result = ParallelReasoner(width=2).run(domain, None, expansion_budget=2)
    assert result.graph["X"].verified_dead
    assert not result.graph["A"].verified_dead
    assert not result.graph["S"].verified_dead


def test_countdown_witness_replays_to_a_valid_expression():
    problem = {"numbers": [24, 98, 19, 3], "target": 361}
    can_solve = solver(problem["target"])
    judge = lambda states, target: [float(can_solve(state)) for state in states]
    domain = CountdownParallelDomain(judge)
    result = ParallelReasoner(width=6).run(domain, problem, expansion_budget=100)
    assert result.solved and result.expansions <= 100
    value, expression = replay_expression(problem["numbers"], result.witness_actions)
    parsed = parse_expr(expression)
    assert value == 361 and parsed is not None and parsed[0] == 361
    assert sorted(parsed[2]) == sorted(problem["numbers"])
