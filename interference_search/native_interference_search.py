"""Synchronized native Interference Search over executed model proposals.

This keeps the defining round boundary: every live state proposes, every
proposal executes, identical resulting states merge, certified failures feed
the shared refutation context, and only then does the next frontier advance.
"""

import asyncio
import inspect
import math
import time
from dataclasses import dataclass, field

from .async_execution import AsyncAction


@dataclass
class NativeInterferenceResult:
    solved: bool = False
    solution: object = None
    witness_actions: list = field(default_factory=list)
    rounds: int = 0
    expanded: int = 0
    proposed: int = 0
    executions: int = 0
    merged_away: int = 0
    certified_dead: int = 0
    wall_seconds: float = 0.0
    trace: list = field(default_factory=list)
    events: list = field(default_factory=list, repr=False)


async def synchronized_search(domain, problem, expansion_budget: int, width: int = 8,
                              execution_workers: int = 8, merge: bool = True):
    """Run Claude's expand-execute-merge-refute-advance loop.

    ``domain.propose_batch`` must return one list of ``AsyncAction`` objects per
    live state. Action priorities are model scores. When several parents reach
    the same exact key, their positive evidence pools with log-sum-exp while a
    single representative state and witness are retained.
    """
    if min(expansion_budget, width, execution_workers) < 1:
        raise ValueError("budget, width and worker count must be positive")
    started = time.perf_counter()
    result = NativeInterferenceResult()
    context = problem
    if hasattr(domain, "prepare"):
        context = domain.prepare(problem)
        context = await context if inspect.isawaitable(context) else context

    start = domain.start(problem)
    live = [(start, [], 0.0)]
    expanded_keys = set()
    semaphore = asyncio.Semaphore(execution_workers)

    async def execute(parent, item):
        async with semaphore:
            value = domain.execute(parent, item.action, context)
            child = await value if inspect.isawaitable(value) else value
        if hasattr(domain, "observe_prediction") and item.predicted_successor is not None:
            domain.observe_prediction(parent, item.action, item.predicted_successor,
                                      child, context)
        if hasattr(domain, "observe_execution"):
            observed = domain.observe_execution(parent, item.action, child, context)
            if inspect.isawaitable(observed):
                await observed
        return child

    try:
        while live and result.expanded < expansion_budget:
            batch = live[:max(0, expansion_budget - result.expanded)]
            if not batch:
                break
            states = [state for state, _, _ in batch]
            round_context = (domain.snapshot_round(context)
                             if hasattr(domain, "snapshot_round") else None)
            result.rounds += 1
            result.expanded += len(states)
            expanded_keys.update(domain.key(state) for state in states)
            proposals = domain.propose_batch(states, context)
            proposals = await proposals if inspect.isawaitable(proposals) else proposals
            if len(proposals) != len(states):
                raise ValueError("proposer must return one action list per live state")

            pending = []
            metadata = []
            for parent_index, actions in enumerate(proposals):
                for rank, item in enumerate(actions):
                    if not isinstance(item, AsyncAction) or not math.isfinite(item.priority):
                        raise ValueError("proposals must be finite-priority AsyncAction objects")
                    pending.append(execute(states[parent_index], item))
                    metadata.append((parent_index, rank, item))
            children = await asyncio.gather(*pending) if pending else []
            result.proposed += len(children)
            result.executions += len(children)

            pool = {}
            transitions = []
            goal = None
            for child, (parent_index, rank, item) in zip(children, metadata):
                witness = batch[parent_index][1] + [item.action]
                transition = {
                    "parent_index": parent_index, "parent": states[parent_index],
                    "rank": rank, "action": item.action,
                    "priority": float(item.priority), "child": child,
                    "goal": domain.is_goal(child, problem),
                    "certified_dead": domain.certified_dead(child, problem),
                }
                transitions.append(transition)
                if transition["goal"]:
                    result.solved = True
                    result.solution = child
                    result.witness_actions = witness
                    goal = transition
                    continue
                if transition["certified_dead"]:
                    result.certified_dead += 1
                    continue
                exact_key = domain.key(child)
                if merge and exact_key in expanded_keys:
                    result.merged_away += 1
                    continue
                key = exact_key if merge else (len(pool), exact_key)
                rank_vote = 1.0 / (rank + 1)
                if key in pool:
                    entry = pool[key]
                    entry[2].append(float(item.priority))
                    entry[3] += rank_vote
                    result.merged_away += 1
                else:
                    pool[key] = [child, witness, [float(item.priority)], rank_vote]

            ranked = []
            for child, witness, scores, votes in pool.values():
                peak = max(scores)
                pooled = peak + math.log(sum(math.exp(score - peak) for score in scores))
                ranked.append((child, witness, pooled, votes))
            if ranked and hasattr(domain, "score_states"):
                judged = domain.score_states([row[0] for row in ranked], context)
                judged = await judged if inspect.isawaitable(judged) else judged
                if len(judged) != len(ranked) or not all(math.isfinite(x) for x in judged):
                    raise ValueError("state judge must return one finite score per candidate")
                ranked = [(child, witness, pooled + float(judge), votes)
                          for (child, witness, pooled, votes), judge in zip(ranked, judged)]
            ranked.sort(key=lambda row: (row[2], row[3]), reverse=True)
            pruned = [child for child, _, _, _ in ranked[width:]]
            kept = [child for child, _, _, _ in ranked[:width]]
            event = {
                "round": result.rounds,
                "context": round_context,
                "live": states,
                "transitions": transitions,
                "ranked": [child for child, _, _, _ in ranked],
                "kept": kept,
                "pruned": pruned,
                "goal": goal,
            }
            result.events.append(event)
            if hasattr(domain, "observe_round"):
                observed = domain.observe_round(event, context)
                if inspect.isawaitable(observed):
                    await observed
            if goal is not None:
                return result
            if hasattr(domain, "observe_pruned") and pruned:
                observed = domain.observe_pruned(
                    pruned, context)
                if inspect.isawaitable(observed):
                    await observed
            live = [(child, witness, pooled) for child, witness, pooled, _ in ranked[:width]]
            result.trace.append({
                "round": result.rounds,
                "live": len(states),
                "raw": len(children),
                "unique": len(pool),
                "kept": len(live),
                "certified_dead": result.certified_dead,
                "merged_away": result.merged_away,
            })
        return result
    finally:
        result.wall_seconds = time.perf_counter() - started
