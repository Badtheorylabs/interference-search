"""Search strategies compared in the Countdown ablations. All take the same judge and the same cost unit:
one expansion = the judge looks at one state and the environment executes all of its moves.

  linear            one chain at a time: sample the next state by the judge's score, restart on failure
  frontier          level-synchronous frontier over merged states, width = budget / depth
  frontier_nomerge  the same, but duplicate states are kept and occupy slots
  bestfirst         strictly sequential, a ranked memory of every state seen, always continue from the best
"""
import heapq
import math

import torch

from interference_search.countdown import children, moves


def is_goal(state, target):
    return state == (target,)


def linear(judge, numbers, target, budget, rng, refute_tau=None, temp=0.5):
    """Returns (solved, expansions used, sequential rounds)."""
    start, used = tuple(sorted(numbers)), 0
    while used < budget:
        s = start
        while len(s) > 1 and used < budget:
            used += 1
            ch = children(s)
            if any(is_goal(c, target) for c in ch):
                return True, used, used
            ch = [c for c in ch if len(c) > 1]
            if not ch:
                break
            v = torch.tensor(judge(ch, target))
            if refute_tau is not None:
                keep = v >= refute_tau
                if not keep.any():
                    break                      # every continuation judged dead: restart
                ch = [c for c, k in zip(ch, keep.tolist()) if k]
                v = v[keep]
            p = torch.softmax(torch.log(v.clamp(1e-6)) / temp, 0)
            s = ch[torch.multinomial(p, 1, generator=rng).item()]
    return False, used, used


def frontier(judge, numbers, target, budget, refute_tau=None, merge=True):
    width = max(1, budget // (len(numbers) - 1))  # spend the budget in parallel
    live, used, rounds = [tuple(sorted(numbers))], 0, 0
    while live and used < budget:
        rounds += 1
        nxt = set() if merge else []
        for s in live:
            if used >= budget:
                break
            used += 1
            for c in (moves(s) if not merge else children(s)):
                if is_goal(c, target):
                    return True, used, rounds
                if len(c) > 1:
                    nxt.add(c) if merge else nxt.append(c)   # merging happens here
        cand = sorted(nxt) if merge else nxt
        v = judge(cand, target)
        if merge:
            scored = sorted(zip(v, cand), reverse=True)            # ties broken by state
        else:
            scored = [(v[i], cand[i]) for i in sorted(range(len(cand)), key=lambda i: -v[i])]
        if refute_tau is not None:
            scored = [(x, c) for x, c in scored if x >= refute_tau]
        live = [c for _, c in scored[:width]]
    return False, used, rounds


def bestfirst(judge, numbers, target, budget, merge=True):
    start = tuple(sorted(numbers))
    heap, seen, used, tie = [(-1.0, 0, start)], {start}, 0, 1
    while heap and used < budget:
        _, _, s = heapq.heappop(heap)
        used += 1
        ch = children(s) if merge else moves(s)
        if any(is_goal(c, target) for c in ch):
            return True, used, used
        ch = [c for c in ch if len(c) > 1 and (not merge or c not in seen)]
        for c, v in zip(ch, judge(ch, target)):
            if merge:
                seen.add(c)
            heapq.heappush(heap, (-v, tie, c))
            tie += 1
    return False, used, used
