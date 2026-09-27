import asyncio

from interference_search.async_execution import AsyncAction
from interference_search.native_interference_search import synchronized_search


class ExactToyDomain:
    def __init__(self):
        self.refutations = []

    def start(self, problem):
        return (0, "root")

    def key(self, state):
        return state[0]

    async def propose_batch(self, states, context):
        table = {
            0: [AsyncAction("a", priority=2.0), AsyncAction("b", priority=1.0)],
            1: [AsyncAction("merge", priority=2.0), AsyncAction("dead", priority=1.0)],
            2: [AsyncAction("merge", priority=1.5)],
            3: [AsyncAction("goal", priority=3.0)],
        }
        return [table.get(state[0], []) for state in states]

    async def execute(self, state, action, context):
        return {
            (0, "a"): (1, "left"), (0, "b"): (2, "right"),
            (1, "merge"): (3, "joined-a"), (2, "merge"): (3, "joined-b"),
            (1, "dead"): (-1, "failed"), (3, "goal"): (4, "answer"),
        }[(state[0], action)]

    def observe_execution(self, state, action, child, context):
        if child[0] == -1:
            self.refutations.append(child)

    def is_goal(self, state, problem):
        return state[0] == 4

    def certified_dead(self, state, problem):
        return state[0] == -1


def test_rounds_execute_merge_refute_then_advance_together():
    domain = ExactToyDomain()
    result = asyncio.run(synchronized_search(
        domain, {}, expansion_budget=8, width=4, execution_workers=4))
    assert result.solved
    assert result.witness_actions == ["a", "merge", "goal"]
    assert result.rounds == 3
    assert result.merged_away == 1
    assert result.certified_dead == 1
    assert domain.refutations == [(-1, "failed")]
    assert result.trace[1]["raw"] == 3
    assert result.trace[1]["unique"] == 1
    assert result.trace[1]["kept"] == 1


def test_disabling_merge_keeps_convergent_paths_separate():
    result = asyncio.run(synchronized_search(
        ExactToyDomain(), {}, expansion_budget=4, width=4,
        execution_workers=4, merge=False))
    assert result.merged_away == 0
    assert result.trace[1]["unique"] == 2
