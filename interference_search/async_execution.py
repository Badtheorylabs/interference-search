"""Event-driven parallel search with concurrent execution and exact sharing.

The model proposes actions in batches. Tools execute concurrently. A finished
tool call can release its child for the next model batch while sibling calls
are still running. Exact state keys merge paths; optional exact execution keys
coalesce identical pending or completed tool calls. Neither key is a soft
behavioral similarity claim.
"""

import asyncio
import inspect
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Hashable


@dataclass(frozen=True)
class AsyncAction:
    action: Any
    predicted_successor: Any = None
    priority: float = 0.0


@dataclass
class AsyncNode:
    state: Any
    key: Hashable
    depth: int
    witness_parent: Hashable | None = None
    witness_action: Any = None
    expanded: bool = False
    certified_dead: bool = False
    priority: float = 0.0


@dataclass
class ExecutionJob:
    key: Hashable | None
    state: Any
    action: Any
    waiters: list[tuple[Hashable, AsyncAction]]
    priority: float = 0.0


@dataclass
class AsyncSearchResult:
    solved: bool = False
    solution: Any = None
    witness_actions: list[Any] = field(default_factory=list)
    expansions: int = 0
    proposal_batches: int = 0
    proposed_actions: int = 0
    execution_requests: int = 0
    execution_calls: int = 0
    execution_coalesced: int = 0
    execution_cache_hits: int = 0
    state_merges: int = 0
    peak_inflight_executions: int = 0
    peak_graph_nodes: int = 0
    cancelled_executions: int = 0
    prepare_calls: int = 0
    observations: int = 0
    predictions_recorded: int = 0
    wall_seconds: float = 0.0
    graph: dict[Hashable, AsyncNode] = field(default_factory=dict, repr=False)


class AsyncExecutionSearch:
    """Search an async domain under explicit model, tool, and expansion limits.

    Required domain methods: start, key, propose_batch, execute, is_goal,
    certified_dead. Optional ``prepare(problem)`` builds one shared model
    context for all proposals and executions. Optional ``observe_execution``
    feeds verified tool results back into it. The async proposer returns one
    list of ``AsyncAction`` per input state. ``execution_key`` is optional and
    must prove identical tool outcomes whenever two calls share a key.
    """

    def __init__(self, model_batch_size: int = 8, execution_workers: int = 8,
                 mode: str = "pipeline", merge_states: bool = True,
                 dedupe_executions: bool = True, batch_wait_seconds: float = 0.0):
        if model_batch_size < 1 or execution_workers < 1:
            raise ValueError("batch size and worker count must be positive")
        if mode not in {"pipeline", "barrier"} or batch_wait_seconds < 0:
            raise ValueError("invalid scheduler mode or batch wait")
        self.model_batch_size = model_batch_size
        self.execution_workers = execution_workers
        self.mode = mode
        self.merge_states = merge_states
        self.dedupe_executions = dedupe_executions
        self.batch_wait_seconds = batch_wait_seconds

    async def run(self, domain, problem, expansion_budget: int) -> AsyncSearchResult:
        if expansion_budget < 1:
            raise ValueError("expansion budget must be positive")
        started = time.perf_counter()
        result = AsyncSearchResult()
        context = problem
        if hasattr(domain, "prepare"):
            prepared = domain.prepare(problem)
            context = await prepared if inspect.isawaitable(prepared) else prepared
            result.prepare_calls = 1
        next_id = 0

        def node_key(state):
            nonlocal next_id
            key = domain.key(state)
            if self.merge_states:
                return key
            next_id += 1
            return (next_id, key)

        start_state = domain.start(problem)
        start_key = node_key(start_state)
        nodes = {start_key: AsyncNode(start_state, start_key, depth=0)}
        ready = deque([start_key])
        queued_jobs: list[ExecutionJob] = []
        executing: dict[asyncio.Task, ExecutionJob] = {}
        active_by_key: dict[Hashable, ExecutionJob] = {}
        execution_cache: dict[Hashable, Any] = {}
        model_task: asyncio.Task | None = None
        model_batch: list[Hashable] = []
        wait_task: asyncio.Task | None = None
        batch_wait_expired = False

        def witness(key):
            actions = []
            while key != start_key:
                node = nodes[key]
                actions.append(node.witness_action)
                key = node.witness_parent
            return list(reversed(actions))

        def accept_result(parent_key, item, child_state):
            action = item.action
            child_key = domain.key(child_state) if self.merge_states else node_key(child_state)
            parent_depth = nodes[parent_key].depth
            child = nodes.get(child_key)
            if child is None:
                child = AsyncNode(child_state, child_key, parent_depth + 1,
                                  witness_parent=parent_key, witness_action=action,
                                  priority=item.priority)
                nodes[child_key] = child
                result.peak_graph_nodes = max(result.peak_graph_nodes, len(nodes))
            else:
                if child.depth != parent_depth + 1:
                    raise ValueError("merged states reached at different depths; include depth in key")
                result.state_merges += 1
                if not child.expanded:
                    child.priority = max(child.priority, item.priority)
            if domain.is_goal(child_state, problem):
                result.solved = True
                result.solution = child_state
                result.witness_actions = witness(child_key)
            elif not child.expanded and not child.certified_dead and child_key not in ready:
                if domain.certified_dead(child_state, problem):
                    child.certified_dead = True
                else:
                    ready.append(child_key)

        def record_prediction(parent_key, item, child_state):
            if item.predicted_successor is None or not hasattr(domain, "observe_prediction"):
                return
            domain.observe_prediction(nodes[parent_key].state, item.action,
                                      item.predicted_successor, child_state, context)
            result.predictions_recorded += 1

        def accept_job(job, child_state):
            for parent_key, item in job.waiters:
                record_prediction(parent_key, item, child_state)
                accept_result(parent_key, item, child_state)

        def add_job(parent_key, item):
            action = item.action
            if not math.isfinite(item.priority):
                raise ValueError("action priority must be finite")
            result.execution_requests += 1
            key = None
            if self.dedupe_executions and hasattr(domain, "execution_key"):
                key = domain.execution_key(nodes[parent_key].state, action, context)
            if key is not None:
                if key in execution_cache:
                    result.execution_cache_hits += 1
                    record_prediction(parent_key, item, execution_cache[key])
                    accept_result(parent_key, item, execution_cache[key])
                    return
                active = active_by_key.get(key)
                if active is not None:
                    active.waiters.append((parent_key, item))
                    active.priority = max(active.priority, item.priority)
                    result.execution_coalesced += 1
                    return
            job = ExecutionJob(key, nodes[parent_key].state, action,
                               [(parent_key, item)], item.priority)
            queued_jobs.append(job)
            if key is not None:
                active_by_key[key] = job

        async def execute(job):
            value = domain.execute(job.state, job.action, context)
            return await value if inspect.isawaitable(value) else value

        async def propose(states):
            value = domain.propose_batch(states, context)
            return await value if inspect.isawaitable(value) else value

        def start_execution_jobs():
            while queued_jobs and len(executing) < self.execution_workers and not result.solved:
                best = max(range(len(queued_jobs)),
                           key=lambda index: queued_jobs[index].priority)
                job = queued_jobs.pop(best)
                task = asyncio.create_task(execute(job))
                executing[task] = job
                result.execution_calls += 1
                result.peak_inflight_executions = max(
                    result.peak_inflight_executions, len(executing))

        try:
            result.peak_graph_nodes = 1
            while True:
                start_execution_jobs()
                can_model = (model_task is None and ready and
                             result.expansions < expansion_budget and
                             (self.mode == "pipeline" or not queued_jobs and not executing))
                if can_model:
                    should_wait = (self.batch_wait_seconds > 0 and
                                   len(ready) < self.model_batch_size and executing and
                                   self.mode == "pipeline")
                    if should_wait and wait_task is None and not batch_wait_expired:
                        wait_task = asyncio.create_task(asyncio.sleep(self.batch_wait_seconds))
                    if not should_wait or batch_wait_expired:
                        if wait_task is not None:
                            wait_task.cancel()
                            wait_task = None
                        batch_wait_expired = False
                        count = min(len(ready), self.model_batch_size,
                                    expansion_budget - result.expansions)
                        ordered = sorted(ready, key=lambda key: nodes[key].priority,
                                         reverse=True)
                        model_batch = ordered[:count]
                        selected = set(model_batch)
                        ready = deque(key for key in ready if key not in selected)
                        for key in model_batch:
                            nodes[key].expanded = True
                        result.expansions += count
                        result.proposal_batches += 1
                        model_task = asyncio.create_task(propose([nodes[key].state
                                                                  for key in model_batch]))
                pending = set(executing)
                if model_task is not None:
                    pending.add(model_task)
                if wait_task is not None:
                    pending.add(wait_task)
                if not pending or result.solved:
                    break
                completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if model_task in completed:
                    proposals = model_task.result()
                    if len(proposals) != len(model_batch):
                        raise ValueError("proposer must return one action list per state")
                    for parent_key, actions in zip(model_batch, proposals):
                        for item in actions:
                            if not isinstance(item, AsyncAction):
                                raise ValueError("proposer returned a non-AsyncAction")
                            result.proposed_actions += 1
                            add_job(parent_key, item)
                    model_task = None
                    model_batch = []
                if wait_task in completed:
                    wait_task = None
                    batch_wait_expired = True
                for task in completed:
                    job = executing.pop(task, None)
                    if job is None:
                        continue
                    child_state = task.result()
                    if hasattr(domain, "observe_execution"):
                        observed = domain.observe_execution(job.state, job.action,
                                                            child_state, context)
                        if inspect.isawaitable(observed):
                            await observed
                        result.observations += 1
                    if job.key is not None:
                        active_by_key.pop(job.key)
                        execution_cache[job.key] = child_state
                    accept_job(job, child_state)
        finally:
            remaining = list(executing)
            if model_task is not None:
                remaining.append(model_task)
            if wait_task is not None:
                remaining.append(wait_task)
            result.cancelled_executions = sum(not task.done() for task in executing)
            for task in remaining:
                task.cancel()
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)
            result.wall_seconds = time.perf_counter() - started
            result.graph = nodes
        return result
