"""Interference Search: one loop, pluggable domains.

The search works on explicit states. Each round, every live state is expanded together and the
environment executes the proposed moves. States with the same merge key collapse into one, and the
votes of their parents add up. A judge ranks what is left, and the best `width` states advance to
the next round. The only things carried between rounds are the live states and the set of keys
already expanded.

A domain supplies six functions:

    start(problem)                  -> initial state
    propose(states, problem)        -> for each state, a list of candidate next states, plus the
                                       generation cost (tokens) spent proposing them
    key(state)                      -> hashable merge key; equal keys mean "the same situation"
    judge(states, problem)          -> score per state (higher = more promising), plus cost;
                                       return None as the scores to rank by pooled votes instead
    is_goal(state, problem)         -> bool
    is_dead(state, problem)         -> bool: the state is finished and wrong

Cost is counted in the domain's own unit, such as generated tokens or judged states, and the loop
stops when the budget runs out.
"""
import math
import random
from dataclasses import dataclass, field


@dataclass
class Result:
    solved: bool
    solution: object = None
    cost: float = 0.0
    rounds: int = 0
    expanded: int = 0
    merged_away: int = 0
    trace: list = field(default_factory=list)


def search(domain, problem, budget, width=6, merge=True, restarts=True):
    live = [domain.start(problem)]
    expanded_keys = set()
    res = Result(solved=False)
    while res.cost < budget:
        if not live:
            if not restarts:
                break
            live = [domain.start(problem)]
        res.rounds += 1
        proposals, pcost = domain.propose(live, problem)
        res.cost += pcost
        res.expanded += len(live)
        for s in live:
            expanded_keys.add(domain.key(s))
        pool = {}
        raw = 0
        for kids in proposals:
            for rank, c in enumerate(kids):
                raw += 1
                if domain.is_goal(c, problem):
                    res.solved, res.solution = True, c
                    return res
                if domain.is_dead(c, problem):
                    continue
                k = domain.key(c) if merge else (domain.key(c), raw)
                if merge and k in expanded_keys:
                    continue
                if k in pool:
                    pool[k][1] += 1.0 / (rank + 1)       # merged: votes from several parents pool
                else:
                    pool[k] = [c, 1.0 / (rank + 1)]
        res.merged_away += raw - len(pool)
        if not pool:
            live = []
            continue
        cands = [v[0] for v in pool.values()]
        votes = [v[1] for v in pool.values()]
        scores, jcost = domain.judge(cands, problem)
        res.cost += jcost
        if scores is None:
            scores = votes
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        live = [cands[i] for i in order[:width]]
        res.trace.append({"round": res.rounds, "pool": len(pool), "raw": raw, "kept": len(live)})
    return res


def linear_search(domain, problem, budget, temp=0.5, rng=None):
    """The baseline: one chain at a time. Judge the children of the current state, sample one and
    continue; when the chain dies, restart from the beginning. Same domain, judge and cost unit."""
    rng = rng or random.Random(0)
    res = Result(solved=False)
    while res.cost < budget:
        s = domain.start(problem)
        while res.cost < budget:
            res.rounds += 1
            res.expanded += 1
            (kids,), pcost = domain.propose([s], problem)
            res.cost += pcost
            for c in kids:
                if domain.is_goal(c, problem):
                    res.solved, res.solution = True, c
                    return res
            kids = [c for c in kids if not domain.is_dead(c, problem)]
            if not kids:
                break
            scores, jcost = domain.judge(kids, problem)
            res.cost += jcost
            if scores is None:
                scores = [1.0 / (i + 1) for i in range(len(kids))]
            w = [math.exp(math.log(max(x, 1e-6)) / temp) for x in scores]
            s = rng.choices(kids, weights=w)[0]
    return res
