"""Freeze 100 rare-solution six-number problems before judge evaluation."""
import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

from interference_search.countdown import gen_problem,moves
from experiments.submission.common import identity,write_json,digest


def capped_paths(problem, cap=24):
    @lru_cache(None)
    def count(state):
        if len(state)==1:
            return int(state[0]==problem["target"])
        total=0
        for child in moves(state):
            total+=count(child)
            if total>cap:
                return cap+1
        return total
    return count(tuple(sorted(problem["numbers"])))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--max-attempts",type=int,default=20000)
    args=ap.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    parent=json.loads(Path(args.parent).read_text())
    forbidden={identity(p) for group in ("main","generalisation","training_sources") for p in parent[group]}
    rng=random.Random(args.seed)
    problems=[]
    for attempt in range(args.max_attempts):
        problem=gen_problem(rng,n_numbers=6)
        ident=identity(problem)
        if ident in forbidden:
            continue
        count=capped_paths(problem)
        if 1<=count<=24:
            problem.update(problem_id=ident,n_numbers=6,cohort="hard",solution_paths=count)
            problems.append(problem)
            forbidden.add(ident)
            print(f"rare six-number {len(problems)}/{args.n}; attempts {attempt+1}; paths {count}",flush=True)
            if len(problems)==args.n:
                write_json(args.out,{"version":1,"seed":args.seed,"parent_sha256":digest(args.parent),
                           "generalisation":problems,"main":[],"attempts":attempt+1,"selection":"exact solution operation paths <=24; no judge score consulted"})
                return
    raise RuntimeError(f"only {len(problems)} hard problems after {args.max_attempts} attempts")


if __name__=="__main__":
    main()
