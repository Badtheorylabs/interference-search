"""Interactive candidate policies for the controlled register-program domain.

The compact BDD and cached-factor policies receive the same verifier feedback.
The factor policy is the simple deterministic baseline; the blacklist policy
uses failures only to avoid exact retries. Candidate batches are selected
before any verifier result from that batch is applied.
"""

from dataclasses import dataclass

from .quotient_state import QuotientState, VerifiedRefutation
from .register_program import Counterexample, PublicRegisterGrammar


@dataclass
class PolicyStats:
    selection_nodes: int = 0
    certificate_patterns: int = 0
    forbidden_cubes_applied: int = 0
    peak_compact_nodes: int = 0
    represented_after_last_update: int | None = None


class FullCandidateBlacklist:
    def __init__(self, grammar: PublicRegisterGrammar):
        self.task = grammar
        self.next_value = 0
        self.seen: set[tuple[int, ...]] = set()
        self.stats = PolicyStats()

    def batch(self, count: int, preferences=None):
        result = []
        if preferences is not None:
            for preferred in preferences[:count]:
                if preferred not in self.seen:
                    result.append(preferred)
            return result
        while len(result) < count and self.next_value < (1 << self.task.decisions):
            value = self.next_value
            self.next_value += 1
            result.append(tuple((value >> (self.task.decisions - index - 1)) & 1
                                for index in range(self.task.decisions)))
            self.seen.add(result[-1])
        return result

    def update(self, feedback: Counterexample):
        return None


class CachedFactorSolver:
    def __init__(self, grammar: PublicRegisterGrammar, node_cap: int = 1_000_000):
        self.task = grammar
        self.node_cap = node_cap
        self.factors: dict[int, tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = {}
        self.stats = PolicyStats()

    def update(self, feedback: Counterexample):
        if feedback.register in self.factors:
            return
        allowed = self.task.allowed_patterns(feedback)
        self.stats.certificate_patterns += 1 << len(feedback.scope)
        self.factors[feedback.register] = (feedback.scope, allowed)

    def _compatible(self, assignments: list[int]) -> bool:
        for scope, patterns in self.factors.values():
            if not any(all(assignments[index] < 0 or assignments[index] == bit
                           for index, bit in zip(scope, pattern)) for pattern in patterns):
                return False
        return True

    def _recover(self, excluded: set[tuple[int, ...]], preferred=None):
        assignment = [-1] * self.task.decisions
        nodes_this_call = 0

        def visit(index: int):
            nonlocal nodes_this_call
            nodes_this_call += 1
            self.stats.selection_nodes += 1
            if nodes_this_call > self.node_cap:
                raise RuntimeError("factor solver node cap reached")
            if not self._compatible(assignment):
                return None
            if index == self.task.decisions:
                candidate = tuple(assignment)
                return None if candidate in excluded else candidate
            for bit in ((0, 1) if preferred is None or preferred[index] == 0
                        else (1, 0)):
                assignment[index] = bit
                answer = visit(index + 1)
                if answer is not None:
                    assignment[index] = -1
                    return answer
            assignment[index] = -1
            return None

        return visit(0)

    def batch(self, count: int, preferences=None):
        selected = []
        excluded = set()
        for position in range(count):
            candidate = self._recover(excluded,
                                      None if preferences is None else preferences[position])
            if candidate is None:
                break
            selected.append(candidate)
            excluded.add(candidate)
        return selected


class CompactQuotientSolver:
    def __init__(self, grammar: PublicRegisterGrammar):
        self.task = grammar
        self.state = QuotientState(grammar.decisions)
        self.seen_registers: set[int] = set()
        self.stats = PolicyStats(represented_after_last_update=1 << grammar.decisions)

    def update(self, feedback: Counterexample):
        if feedback.register in self.seen_registers:
            return
        self.seen_registers.add(feedback.register)
        allowed = set(self.task.allowed_patterns(feedback))
        scope = feedback.scope
        self.stats.certificate_patterns += 1 << len(scope)
        for value in range(1 << len(scope)):
            pattern = tuple((value >> (len(scope) - index - 1)) & 1
                            for index in range(len(scope)))
            if pattern in allowed:
                continue
            certificate = VerifiedRefutation.from_mapping(dict(zip(scope, pattern)))
            self.state.refute(
                certificate,
                lambda item, bindings=certificate.bindings, values=pattern:
                item.bindings == bindings and self.task.certify_forbidden(feedback, values),
            )
            self.stats.forbidden_cubes_applied += 1
        self.stats.peak_compact_nodes = max(self.stats.peak_compact_nodes,
                                            self.state.reachable_nodes())
        self.stats.represented_after_last_update = self.state.count()

    def batch(self, count: int, preferences=None):
        fork = self.state.fork()
        selected = []
        for position in range(count):
            candidate = fork.recover(None if preferences is None else preferences[position])
            if candidate is None:
                break
            selected.append(candidate)
            fork.exclude_candidate_for_selection(candidate)
        return selected
