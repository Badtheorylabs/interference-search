"""Countdown as an Interference Search domain.

The environment lists every legal move for free, so proposing costs nothing and the judge does all the
work; cost is counted in states judged.
"""
from .countdown import moves


class Countdown:
    def __init__(self, judge):
        self.judge_fn = judge                   # judge(states, target) -> scores

    def start(self, p):
        return tuple(sorted(p["numbers"]))

    def propose(self, states, p):
        return [sorted(set(moves(s))) for s in states], 0

    def key(self, s):
        return s

    def judge(self, states, p):
        return self.judge_fn(states, p["target"]), len(states)

    def is_goal(self, s, p):
        return s == (p["target"],)

    def is_dead(self, s, p):
        return len(s) == 1 and s[0] != p["target"]
