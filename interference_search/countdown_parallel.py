"""Countdown adapter for the shared parallel-reasoning graph."""

import math

from .countdown_actions import CountdownActionCodec
from .judge import hand_alive
from .parallel_reasoning import Transition


class CountdownParallelDomain:
    def __init__(self, judge, max_numbers: int = 7, sound_bound: bool = False):
        self.judge_fn = judge
        self.codec = CountdownActionCodec(max_numbers)
        self.sound_bound = sound_bound

    def start(self, problem):
        return tuple(sorted(problem["numbers"]))

    def key(self, state):
        return state

    def propose_batch(self, states, problem):
        batches = []
        for state in states:
            legal = [index for index, allowed in enumerate(self.codec.legal_mask(state)) if allowed]
            log_prior = -math.log(len(legal)) if legal else 0.0
            batches.append([
                Transition(self.codec.apply(state, index), self.codec.actions[index], log_prior)
                for index in legal
            ])
        return batches

    def judge(self, states, problem):
        return self.judge_fn(states, problem["target"])

    def is_goal(self, state, problem):
        return state == (problem["target"],)

    def certified_dead(self, state, problem):
        if len(state) == 1:
            return state[0] != problem["target"]
        return self.sound_bound and not hand_alive(state, problem["target"])


def replay_expression(numbers, actions):
    """Turn a model-independent action witness into a legal expression."""
    items = sorted((number, str(number)) for number in numbers)
    for action in actions:
        i, j = action.left, action.right
        if i >= len(items) or j >= len(items) or i == j:
            raise ValueError("witness action points outside the live state")
        a, left = items[i]
        b, right = items[j]
        if action.op == "+":
            value = a + b
        elif action.op == "-":
            value = a - b
        elif action.op == "*":
            value = a * b
        else:
            if b == 0 or a % b:
                raise ValueError("witness division is not exact")
            value = a // b
        if value <= 0:
            raise ValueError("witness has a nonpositive intermediate")
        rest = [item for index, item in enumerate(items) if index not in (i, j)]
        items = sorted(rest + [(value, f"({left} {action.op} {right})")], key=lambda item: item[0])
    if len(items) != 1:
        raise ValueError("witness did not combine every number")
    return items[0]
