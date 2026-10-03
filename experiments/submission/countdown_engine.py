"""Strictly charged search, independent merge/judge/synchrony switches.

One state-work unit is an expansion, a proposed successor, or a judge score.
This is a reproducible abstract work measure, not a FLOP or latency claim.
Offline score replay is labelled by the driver. Every queried score is charged
even when its frozen value is read from a replay table.
"""
import heapq
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass

from interference_search.countdown import moves
from interference_search.judge import hand_alive


@dataclass(frozen=True)
class Method:
    name: str
    merge: bool
    judge: bool
    synchronous: bool
    scheduler: str = "frontier"
    prune: bool = True


def methods():
    out = [Method(f"factorial_m{m}j{j}s{s}", bool(m), bool(j), bool(s))
           for m in (0, 1) for j in (0, 1) for s in (0, 1)]
    out += [Method("single_line", False, True, False, "linear", False),
            Method("bfs", True, False, False, "bfs", False),
            Method("beam", True, True, True, "frontier", False),
            Method("best_first", True, True, False, "bestfirst", False),
            Method("independent_parallel", False, True, True, "parallel", False),
            Method("frontier_without_merging", False, True, True),
            Method("merging_without_learned_judge", True, False, True),
            Method("full_interference_search", True, True, True),
            Method("hand_bound_frontier", True, False, True, "hand", True)]
    return out


def run(problem, method, seed, budget, score, width=16, threshold=0.2,
        budget_unit="state_work"):
    if budget < 0 or width < 1 or budget_unit not in ("state_work", "proposed"):
        raise ValueError("invalid search settings")
    started = time.perf_counter()
    rng = random.Random(seed)
    start = tuple(sorted(problem["numbers"]))
    counters = Counter(states_proposed=0, states_merged=0, states_judged=0,
                       states_pruned=0, states_expanded=0, sequential_rounds=0,
                       terminal_failures=0, frontier_dropped=0, work_used=0)
    depth_rows = {}
    visited = {start} if method.merge else set()
    tie = 0
    live = [(start, [start])]
    heap = []
    queue = deque()
    solved, solution, truncated = start == (problem["target"],), None, False

    def charge(kind):
        cost = 1 if budget_unit == "state_work" or kind == "states_proposed" else 0
        if counters["work_used"] + cost > budget:
            return False
        counters[kind] += 1
        counters["work_used"] += cost
        return True

    def select(candidates, count):
        if method.scheduler in ("linear", "parallel"):
            selected = []
            while candidates and len(selected) < count:
                weights = [max(c[0], 1e-8) ** 2 for c in candidates]
                idx = rng.choices(range(len(candidates)), weights=weights)[0]
                selected.append(candidates.pop(idx))
            return selected
        return sorted(candidates, key=lambda c: (-c[0], c[1]))[:count]

    while live and not solved and counters["work_used"] < budget:
        counters["sequential_rounds"] += 1
        candidates = []
        # Each synchronous parent sees the previous round only. No sibling
        # is expanded from a candidate produced in this round.
        for state, path in live:
            if not charge("states_expanded"):
                truncated = True
                break
            raw = moves(state)
            rng.shuffle(raw)
            eligible = []
            for child in raw:
                if not charge("states_proposed"):
                    truncated = True
                    break
                depth = len(start) - len(child)
                row = depth_rows.setdefault(depth, {"depth": depth, "raw_candidates": 0,
                                                     "states": set(), "merged": 0,
                                                     "judged": 0, "pruned": 0})
                row["raw_candidates"] += 1
                row["states"].add(child)
                if method.merge and child in visited:
                    counters["states_merged"] += 1
                    row["merged"] += 1
                    continue
                if method.merge:
                    visited.add(child)
                if child == (problem["target"],):
                    solved, solution = True, path + [child]
                    break
                if len(child) == 1:
                    counters["terminal_failures"] += 1
                    continue
                eligible.append((child, path + [child]))
            # Score the admitted prefix, bounded before inference. Goal
            # checks and all proposals are charged, including duplicates.
            if solved:
                break
            admitted = []
            for child, child_path in eligible:
                if method.judge and not charge("states_judged"):
                    truncated = True
                    break
                admitted.append((child, child_path))
            values = score([c for c, _ in admitted], problem["target"]) if method.judge else [rng.random() for _ in admitted]
            for (child, child_path), value in zip(admitted, values):
                row = depth_rows[len(start) - len(child)]
                row["judged"] += int(method.judge)
                if ((method.judge and method.prune and value < threshold)
                        or (method.scheduler == "hand" and not hand_alive(child, problem["target"]))):
                    counters["states_pruned"] += 1
                    row["pruned"] += 1
                    continue
                tie += 1
                candidates.append((value, tie, child, child_path))
            if truncated:
                break
        if solved or truncated:
            break
        if method.scheduler == "parallel":
            # Independent lanes choose one child per parent, with no shared
            # visited set and no comparison of children across parents.
            by_parent = {}
            for candidate in candidates:
                by_parent.setdefault(tuple(candidate[3][:-1]), []).append(candidate)
            selected = []
            for group in by_parent.values():
                selected += select(group, width if len(live) == 1 and live[0][0] == start else 1)
            counters["frontier_dropped"] += len(candidates) - len(selected)
            live = [(c[2], c[3]) for c in selected]
            if not live:
                live = [(start, [start])]
        elif method.scheduler == "linear":
            selected = select(candidates.copy(), 1)
            counters["frontier_dropped"] += len(candidates) - len(selected)
            live = [(c[2], c[3]) for c in selected] or [(start, [start])]
        elif method.synchronous:
            selected = select(candidates.copy(), width)
            counters["frontier_dropped"] += len(candidates) - len(selected)
            live = [(c[2], c[3]) for c in selected]
        elif method.scheduler == "bfs":
            queue.extend((c[2], c[3]) for c in candidates)
            live = [queue.popleft()] if queue else []
        else:
            for value, order, child, child_path in candidates:
                heapq.heappush(heap, (-value, order, child, child_path))
            if method.scheduler != "bestfirst" and len(heap) > width:
                counters["frontier_dropped"] += len(heap) - width
                heap = heapq.nsmallest(width, heap)
                heapq.heapify(heap)
            if heap:
                _, _, child, child_path = heapq.heappop(heap)
                live = [(child, child_path)]
            else:
                live = []
    traces = [{**{k: v for k, v in row.items() if k != "states"},
               "unique_states": len(row["states"])} for _, row in sorted(depth_rows.items())]
    return {**dict(counters), "solved": solved, "solution_state_path": solution,
            "failure_reason": None if solved else "budget" if truncated or counters["work_used"] >= budget else "frontier_exhausted",
            "wall_seconds": time.perf_counter() - started,
            "generated_tokens": 0, "prompt_tokens": 0, "depth_trace": traces,
            "merge": method.merge, "learned_judge": method.judge,
            "synchronous": method.synchronous, "prune": method.prune,
            "budget_unit": budget_unit, "width": width, "threshold": threshold}
