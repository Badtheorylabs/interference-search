"""Typed action slots for a model-native Countdown proposer.

The model chooses an action index. This codec checks syntax and executes the
chosen operation; it does not enumerate or rank candidate next states for the
model. Invalid slots can be masked before sampling.
"""

from dataclasses import dataclass


OPS = ("+", "-", "*", "/")


@dataclass(frozen=True)
class CountdownAction:
    left: int
    right: int
    op: str


class CountdownActionCodec:
    def __init__(self, max_numbers: int = 7):
        if max_numbers < 2:
            raise ValueError("max_numbers must be at least two")
        self.max_numbers = max_numbers
        self.actions = tuple(
            CountdownAction(i, j, op)
            for i in range(max_numbers)
            for j in range(max_numbers)
            if i != j
            for op in OPS
        )
        self.index = {action: k for k, action in enumerate(self.actions)}

    def legal(self, state: tuple[int, ...], action_id: int) -> bool:
        action = self.actions[action_id]
        i, j, op = action.left, action.right, action.op
        if i >= len(state) or j >= len(state):
            return False
        a, b = state[i], state[j]
        if op in ("+", "*"):
            return i < j
        if op == "-":
            return a > b
        return b > 1 and a % b == 0

    def legal_mask(self, state: tuple[int, ...]) -> list[bool]:
        return [self.legal(state, k) for k in range(len(self.actions))]

    def apply(self, state: tuple[int, ...], action_id: int) -> tuple[int, ...] | None:
        if not self.legal(state, action_id):
            return None
        action = self.actions[action_id]
        i, j, op = action.left, action.right, action.op
        a, b = state[i], state[j]
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
        elif op == "*":
            value = a * b
        else:
            value = a // b
        rest = [x for k, x in enumerate(state) if k not in (i, j)]
        return tuple(sorted(rest + [value]))
