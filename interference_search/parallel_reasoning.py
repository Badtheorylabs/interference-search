"""Proof-aware, level-synchronous search over one shared state graph.

The proposer is called once for a whole live frontier. Equal *environment*
states share a node, incoming path mass and proof status. A verified dead node
cancels every represented path through it. Scored states outside the live width
remain available for counterfactual recovery instead of restarting at root.

This is a classical reasoning engine. The proposer and judge may be models,
but neither is trained or implied by this module.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Hashable


def logadd(left: float, right: float) -> float:
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    return max(left, right) + math.log1p(math.exp(-abs(left - right)))


@dataclass(frozen=True)
class Transition:
    state: Any
    action: Any
    log_prior: float = 0.0


@dataclass
class SharedNode:
    state: Any
    key: Hashable
    depth: int
    log_mass: float = float("-inf")
    value: float | None = None
    expanded: bool = False
    verified_dead: bool = False
    children: dict[Hashable, float] = field(default_factory=dict)
    parents: set[Hashable] = field(default_factory=set)
    witness_parent: Hashable | None = None
    witness_action: Any = None


@dataclass
class ReasoningResult:
    solved: bool = False
    solution: Any = None
    witness_actions: list[Any] = field(default_factory=list)
    expansions: int = 0
    rounds: int = 0
    proposal_calls: int = 0
    proposal_count: int = 0
    judge_calls: int = 0
    judged_states: int = 0
    merged_proposals: int = 0
    certified_dead_nodes: int = 0
    rescue_rounds: int = 0
    peak_graph_nodes: int = 0
    peak_reserve: int = 0
    wall_seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)
    graph: dict[Hashable, SharedNode] = field(default_factory=dict, repr=False)


class ParallelReasoner:
    """Run a domain with start/key/propose_batch/judge/is_goal/certified_dead."""

    def __init__(self, width: int, mass_weight: float = 0.0, reserve: bool = True):
        if width < 1 or mass_weight < 0:
            raise ValueError("width must be positive and mass_weight nonnegative")
        self.width = width
        self.mass_weight = mass_weight
        self.use_reserve = reserve

    def run(self, domain, problem, expansion_budget: int) -> ReasoningResult:
        if expansion_budget < 1:
            raise ValueError("expansion budget must be positive")
        started = time.perf_counter()
        start = domain.start(problem)
        start_key = domain.key(start)
        nodes = {start_key: SharedNode(start, start_key, depth=0, log_mass=0.0)}
        live = [start_key]
        standby: dict[int, list[Hashable]] = {}
        result = ReasoningResult()
        newly_dead = deque()

        def mark_dead(key: Hashable):
            node = nodes[key]
            if node.verified_dead:
                return
            node.verified_dead = True
            result.certified_dead_nodes += 1
            newly_dead.append(key)

        def propagate_dead_proofs():
            # Run after the whole level has added its edges. A parent cannot be
            # declared dead while some of its legal transitions are pending.
            for key in live:
                node = nodes[key]
                if node.expanded and not node.verified_dead and (
                    not node.children or all(nodes[child].verified_dead for child in node.children)
                ):
                    mark_dead(key)
            while newly_dead:
                dead_key = newly_dead.popleft()
                for parent_key in nodes[dead_key].parents:
                    parent = nodes[parent_key]
                    if parent.expanded and not parent.verified_dead and all(
                        nodes[child].verified_dead for child in parent.children
                    ):
                        mark_dead(parent_key)

        def recompute_masses():
            for node in nodes.values():
                node.log_mass = float("-inf")
            nodes[start_key].log_mass = 0.0
            for node in sorted(nodes.values(), key=lambda item: item.depth):
                if node.log_mass == float("-inf"):
                    continue
                for child_key, edge_mass in node.children.items():
                    child = nodes[child_key]
                    child.log_mass = logadd(child.log_mass, node.log_mass + edge_mass)

        def priority(key: Hashable):
            node = nodes[key]
            value = max(min(node.value if node.value is not None else 0.5, 1 - 1e-6), 1e-6)
            mass_term = self.mass_weight * node.log_mass if self.mass_weight else 0.0
            score = math.log(value / (1 - value)) + mass_term
            return (score, repr(key))

        def rescue_frontier():
            if not self.use_reserve:
                return []
            for depth in sorted(standby, reverse=True):
                pending = [key for key in standby[depth]
                           if not nodes[key].expanded and not nodes[key].verified_dead]
                standby[depth] = []
                if pending:
                    pending.sort(key=priority, reverse=True)
                    chosen = pending[:self.width]
                    standby[depth].extend(pending[self.width:])
                    result.rescue_rounds += 1
                    return chosen
            return []

        while result.expansions < expansion_budget:
            live = [key for key in live if not nodes[key].expanded and not nodes[key].verified_dead]
            if not live:
                live = rescue_frontier()
                if not live:
                    break
            remaining = expansion_budget - result.expansions
            live = live[:remaining]
            depth = nodes[live[0]].depth
            if any(nodes[key].depth != depth for key in live):
                raise ValueError("one batch must contain states at the same depth")
            result.rounds += 1
            result.expansions += len(live)
            result.proposal_calls += 1
            proposed = domain.propose_batch([nodes[key].state for key in live], problem)
            if len(proposed) != len(live):
                raise ValueError("proposer must return one transition list per live state")
            result.proposal_count += sum(len(group) for group in proposed)
            prior_nodes = len(nodes)
            for parent_key, transitions in zip(live, proposed):
                parent = nodes[parent_key]
                parent.expanded = True
                for transition in transitions:
                    if not math.isfinite(transition.log_prior):
                        raise ValueError("transition log prior must be finite")
                    child_key = domain.key(transition.state)
                    child = nodes.get(child_key)
                    if child is None:
                        child = SharedNode(transition.state, child_key, depth=depth + 1,
                                           witness_parent=parent_key, witness_action=transition.action)
                        nodes[child_key] = child
                    elif child.depth != depth + 1:
                        raise ValueError("merge key reached at different depths; include time in state")
                    else:
                        result.merged_proposals += 1
                    child.parents.add(parent_key)
                    parent.children[child_key] = logadd(parent.children.get(child_key, float("-inf")),
                                                        transition.log_prior)
                    if domain.is_goal(transition.state, problem):
                        result.solved = True
                        result.solution = transition.state
                        witness = []
                        cursor = child_key
                        while cursor != start_key:
                            current = nodes[cursor]
                            witness.append(current.witness_action)
                            cursor = current.witness_parent
                        result.witness_actions = list(reversed(witness))
                        if not self.mass_weight:
                            recompute_masses()
                        result.peak_graph_nodes = max(result.peak_graph_nodes, len(nodes))
                        result.graph = nodes
                        result.wall_seconds = time.perf_counter() - started
                        return result
                    if not child.verified_dead and domain.certified_dead(transition.state, problem):
                        mark_dead(child_key)
            propagate_dead_proofs()
            if self.mass_weight:
                recompute_masses()
            candidate_keys = [key for key, node in nodes.items()
                              if node.depth == depth + 1 and not node.expanded
                              and not node.verified_dead and node.value is None]
            if candidate_keys:
                scores = domain.judge([nodes[key].state for key in candidate_keys], problem)
                if len(scores) != len(candidate_keys):
                    raise ValueError("judge must return one score per state")
                result.judge_calls += 1
                result.judged_states += len(candidate_keys)
                for key, score in zip(candidate_keys, scores):
                    if not 0 <= score <= 1 or not math.isfinite(score):
                        raise ValueError("judge scores must be finite probabilities")
                    nodes[key].value = float(score)
            candidates = [key for key, node in nodes.items()
                          if node.depth == depth + 1 and not node.expanded and not node.verified_dead]
            candidates.sort(key=priority, reverse=True)
            live = candidates[:self.width]
            if self.use_reserve:
                existing = [key for key in standby.get(depth + 1, [])
                            if key not in live and not nodes[key].expanded and not nodes[key].verified_dead]
                seen_reserve = set(existing)
                existing.extend(key for key in candidates[self.width:] if key not in seen_reserve)
                standby[depth + 1] = existing
            result.peak_graph_nodes = max(result.peak_graph_nodes, len(nodes))
            result.peak_reserve = max(result.peak_reserve, sum(len(group) for group in standby.values()))
            result.trace.append({
                "round": result.rounds, "depth": depth, "expanded": len(proposed),
                "new_nodes": len(nodes) - prior_nodes, "candidates": len(candidates),
                "live": len(live), "dead_total": result.certified_dead_nodes,
                "reserve": sum(len(group) for group in standby.values()),
            })
        if not self.mass_weight:
            recompute_masses()
        result.wall_seconds = time.perf_counter() - started
        result.graph = nodes
        return result
