"""Exact compact version-space reference for structured search.

This is a reduced ordered binary decision diagram (ROBDD), a classical
baseline and correctness oracle for a future learned quotient state. It can
represent many assignments with few nodes when constraints have reusable
structure. Worst-case size is exponential. A verifier must certify every
subspace exclusion before it is committed.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class VerifiedRefutation:
    """Values of a subset of decisions known to be jointly impossible."""

    bindings: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if any(index < 0 or bit not in (0, 1) for index, bit in self.bindings):
            raise ValueError("decision indices and binary values must be valid")
        if tuple(sorted(self.bindings)) != self.bindings or len({index for index, _ in self.bindings}) != len(self.bindings):
            raise ValueError("refutation bindings must be sorted with unique indices")

    @classmethod
    def from_mapping(cls, values: Mapping[int, int]) -> "VerifiedRefutation":
        bindings = tuple(sorted(values.items()))
        return cls(bindings)


class QuotientState:
    """A canonical DAG for a Boolean hypothesis set.

    Nodes 0 and 1 are false and true. A nonterminal node stores
    ``(decision_index, low_child, high_child)``. Shared suffixes merge through
    the unique table; no complete hypotheses are stored.
    """

    def __init__(self, decisions: int):
        if decisions < 1:
            raise ValueError("decisions must be positive")
        self.decisions = decisions
        self.nodes: list[tuple[int, int, int] | None] = [None, None]
        self.unique: dict[tuple[int, int, int], int] = {}
        self.root = 1
        self.refutations: list[VerifiedRefutation] = []

    def _node(self, var: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = (var, low, high)
        result = self.unique.get(key)
        if result is None:
            result = len(self.nodes)
            self.nodes.append(key)
            self.unique[key] = result
        return result

    def _negate(self, node: int, memo: dict[int, int]) -> int:
        if node <= 1:
            return 1 - node
        if node not in memo:
            var, low, high = self.nodes[node]
            memo[node] = self._node(
                var, self._negate(low, memo), self._negate(high, memo)
            )
        return memo[node]

    def _and(self, left: int, right: int, memo: dict[tuple[int, int], int]) -> int:
        if left == 0 or right == 0:
            return 0
        if left == 1:
            return right
        if right == 1 or left == right:
            return left
        key = (left, right)
        if key not in memo:
            lvar, llo, lhi = self.nodes[left]
            rvar, rlo, rhi = self.nodes[right]
            var = min(lvar, rvar)
            low = self._and(llo if lvar == var else left,
                            rlo if rvar == var else right, memo)
            high = self._and(lhi if lvar == var else left,
                             rhi if rvar == var else right, memo)
            memo[key] = self._node(var, low, high)
        return memo[key]

    def _cube(self, refutation: VerifiedRefutation) -> int:
        node = 1
        for var, bit in reversed(refutation.bindings):
            if not 0 <= var < self.decisions:
                raise ValueError("decision index out of range")
            node = self._node(var, 0, node) if bit else self._node(var, node, 0)
        return node

    def refute(
        self,
        refutation: VerifiedRefutation,
        certify: Callable[[VerifiedRefutation], bool],
    ) -> bool:
        """Remove a certified subspace. Return whether the live set changed."""
        if not refutation.bindings:
            raise ValueError("an empty refutation would erase every hypothesis")
        cube = self._cube(refutation)
        if not certify(refutation):
            raise ValueError("verifier did not certify the proposed subspace")
        new_root = self._and(self.root, self._negate(cube, {}), {})
        changed = new_root != self.root
        self.root = new_root
        self.refutations.append(refutation)
        return changed

    def count(self) -> int:
        @lru_cache(maxsize=None)
        def visit(node: int, level: int) -> int:
            if node == 0:
                return 0
            if node == 1:
                return 1 << (self.decisions - level)
            var, low, high = self.nodes[node]
            gap = var - level
            return (visit(low, var + 1) + visit(high, var + 1)) << gap

        return visit(self.root, 0)

    def contains(self, assignment: tuple[int, ...]) -> bool:
        if len(assignment) != self.decisions or any(bit not in (0, 1) for bit in assignment):
            raise ValueError("assignment has wrong shape or values")
        node = self.root
        while node > 1:
            var, low, high = self.nodes[node]
            node = high if assignment[var] else low
        return node == 1

    def recover(self) -> tuple[int, ...] | None:
        """Return one surviving assignment without replaying prior exclusions."""
        if self.root == 0:
            return None
        answer = [0] * self.decisions
        node = self.root
        while node > 1:
            var, low, high = self.nodes[node]
            if low != 0:
                node = low
            else:
                answer[var] = 1
                node = high
        return tuple(answer)

    def log_partition(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable log total weight of surviving assignments.

        Each assignment has unnormalized log weight ``sum_i logits[i] * bit_i``.
        The DAG handles hard support; a neural model can supply the logits.
        This computes a partition function without enumerating assignments.
        """
        if logits.shape != (self.decisions,):
            raise ValueError("logits must have one value per decision")
        memo: dict[tuple[int, int], torch.Tensor] = {}

        def visit(node: int, level: int) -> torch.Tensor:
            key = (node, level)
            if key in memo:
                return memo[key]
            if node == 0:
                result = logits.new_full((), float("-inf"))
            elif node == 1:
                result = F.softplus(logits[level:]).sum()
            else:
                var, low, high = self.nodes[node]
                skipped = F.softplus(logits[level:var]).sum()
                low_weight = visit(low, var + 1)
                high_weight = logits[var] + visit(high, var + 1)
                result = skipped + torch.logaddexp(low_weight, high_weight)
            memo[key] = result
            return result

        return visit(self.root, 0)

    def best(self, logits: torch.Tensor) -> tuple[int, ...] | None:
        """Highest-weight surviving assignment under independent neural logits."""
        if logits.shape != (self.decisions,):
            raise ValueError("logits must have one value per decision")
        if self.root == 0:
            return None
        scores = [float(x) for x in logits.detach().cpu()]

        @lru_cache(maxsize=None)
        def value(node: int, level: int) -> float:
            if node == 0:
                return float("-inf")
            if node == 1:
                return sum(max(0.0, score) for score in scores[level:])
            var, low, high = self.nodes[node]
            skipped = sum(max(0.0, score) for score in scores[level:var])
            return skipped + max(value(low, var + 1), scores[var] + value(high, var + 1))

        answer = [0] * self.decisions
        node, level = self.root, 0
        while node > 1:
            var, low, high = self.nodes[node]
            for index in range(level, var):
                answer[index] = int(scores[index] > 0)
            if scores[var] + value(high, var + 1) > value(low, var + 1):
                answer[var] = 1
                node = high
            else:
                node = low
            level = var + 1
        for index in range(level, self.decisions):
            answer[index] = int(scores[index] > 0)
        return tuple(answer)

    def reachable_nodes(self) -> int:
        seen = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node <= 1 or node in seen:
                continue
            seen.add(node)
            _, low, high = self.nodes[node]
            stack.extend((low, high))
        return len(seen)
