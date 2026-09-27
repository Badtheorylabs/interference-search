import asyncio

from interference_search.async_execution import AsyncAction, AsyncExecutionSearch


class TimedDomain:
    def __init__(self, edges, tool_delay=None, model_delay=0.002, goal="G"):
        self.edges = edges
        self.tool_delay = tool_delay or {}
        self.model_delay = model_delay
        self.goal = goal

    def start(self, problem):
        return "S"

    def key(self, state):
        return state

    async def propose_batch(self, states, problem):
        await asyncio.sleep(self.model_delay)
        return [[AsyncAction(child) for child in self.edges.get(state, ())]
                for state in states]

    async def execute(self, state, action, problem):
        await asyncio.sleep(self.tool_delay.get((state, action), 0.002))
        return action

    def execution_key(self, state, action, problem):
        return (state, action)

    def is_goal(self, state, problem):
        return state == self.goal

    def certified_dead(self, state, problem):
        return False


def test_pipeline_overlaps_fast_path_with_slow_sibling():
    domain = TimedDomain(
        {"S": ["A", "B"], "A": ["X"], "B": ["C"], "C": ["G"]},
        tool_delay={("S", "A"): 0.08},
    )

    async def compare():
        pipeline = await AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                               mode="pipeline").run(domain, None, 10)
        waited = await AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                             mode="pipeline", batch_wait_seconds=0.001).run(
                                                 domain, None, 10)
        barrier = await AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                              mode="barrier").run(domain, None, 10)
        return pipeline, waited, barrier

    pipeline, waited, barrier = asyncio.run(compare())
    assert pipeline.solved and barrier.solved
    assert pipeline.witness_actions == barrier.witness_actions == ["B", "C", "G"]
    assert pipeline.wall_seconds < barrier.wall_seconds * 0.7
    assert waited.wall_seconds < barrier.wall_seconds * 0.7
    assert pipeline.execution_requests <= barrier.execution_requests
    assert pipeline.peak_inflight_executions == 2
    assert pipeline.cancelled_executions >= 1


def test_exact_inflight_execution_and_state_merging_save_work():
    domain = TimedDomain({"S": ["A", "A"], "A": ["G"]})

    async def compare():
        shared = await AsyncExecutionSearch(model_batch_size=2, execution_workers=2).run(
            domain, None, 10)
        tree = await AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                          merge_states=False,
                                          dedupe_executions=False).run(domain, None, 10)
        return shared, tree

    shared, tree = asyncio.run(compare())
    assert shared.solved and tree.solved
    assert shared.witness_actions == ["A", "G"]
    assert shared.execution_requests == 3
    assert shared.execution_calls == 2
    assert shared.execution_coalesced == 1
    assert shared.state_merges == 1
    assert tree.execution_calls >= 3
    assert shared.expansions <= tree.expansions


def test_expansion_budget_stops_new_model_work_but_drains_running_execution():
    domain = TimedDomain({"S": ["A"], "A": ["G"]})
    result = asyncio.run(AsyncExecutionSearch().run(domain, None, 1))
    assert not result.solved
    assert result.expansions == 1
    assert result.execution_calls == 1
    assert "A" in result.graph


def test_coalesced_execution_keeps_each_native_prediction_pair():
    class PredictedDomain(TimedDomain):
        def __init__(self):
            super().__init__({"S": ["A", "A"]})
            self.events = []

        async def propose_batch(self, states, problem):
            return [[AsyncAction(child, predicted_successor=index)
                     for index, child in enumerate(self.edges.get(state, ()))]
                    for state in states]

        def observe_prediction(self, state, action, prediction, child, context):
            self.events.append((prediction, child))

    domain = PredictedDomain()
    result = asyncio.run(AsyncExecutionSearch().run(domain, None, 1))
    assert result.execution_calls == 1
    assert result.execution_coalesced == 1
    assert result.predictions_recorded == 2
    assert sorted(domain.events) == [(0, "A"), (1, "A")]


def test_native_priority_selects_ready_branch_under_expansion_budget():
    class PriorityDomain(TimedDomain):
        async def propose_batch(self, states, problem):
            return [[AsyncAction(child, priority=10.0 if child == "B" else 0.0)
                     for child in self.edges.get(state, ())] for state in states]

    domain = PriorityDomain({"S": ["A", "B"], "A": ["X"], "B": ["G"]},
                            tool_delay={("S", "A"): 0.001, ("S", "B"): 0.001})
    result = asyncio.run(AsyncExecutionSearch(model_batch_size=1,
                                               execution_workers=2).run(domain, None, 2))
    assert result.solved
    assert result.witness_actions == ["B", "G"]
    assert result.expansions == 2


def test_native_priority_orders_bounded_execution_queue():
    class PriorityDomain(TimedDomain):
        def __init__(self):
            super().__init__({"S": ["A", "B"]}, goal="none")
            self.executed = []

        async def propose_batch(self, states, problem):
            return [[AsyncAction(child, priority=10.0 if child == "B" else 0.0)
                     for child in self.edges.get(state, ())] for state in states]

        async def execute(self, state, action, problem):
            self.executed.append(action)
            return action

    domain = PriorityDomain()
    result = asyncio.run(AsyncExecutionSearch(execution_workers=1).run(domain, None, 1))
    assert domain.executed == ["B", "A"]
    assert result.execution_calls == 2
