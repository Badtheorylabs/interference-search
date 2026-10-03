"""Exact size-transfer and rare-path threshold curves, including failed feasibility attempts."""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from interference_search.countdown import moves
from interference_search.judge import load_judge
from experiments.submission.common import ROOT, append, digest, environment, read_rows, write_json


def graph(problem, state_cap=500000, seconds=120):
    started = time.perf_counter()
    start = tuple(sorted(problem["numbers"]))
    layers = [Counter({start: 1})]
    edges = {}
    traces = []
    total_states = 1
    while len(next(iter(layers[-1]))) > 1:
        nxt = Counter()
        for state, paths in layers[-1].items():
            children = Counter(moves(state))
            edges[state] = children
            for child, count in children.items():
                nxt[child] += paths * count
            if total_states + len(nxt) > state_cap:
                raise RuntimeError(f"exact graph exceeds {state_cap} unique states")
            if time.perf_counter() - started > seconds:
                raise TimeoutError(f"exact graph exceeds {seconds} seconds")
        total_states += len(nxt)
        traces.append({"depth": len(layers), "raw_candidate_paths": sum(nxt.values()),
                       "unique_states": len(nxt),
                       "raw_successor_edges": sum(sum(edges[s].values()) for s in layers[-1])})
        layers.append(nxt)
    alive = {state: state == (problem["target"],) for state in layers[-1]}
    for layer in reversed(layers[:-1]):
        for state in layer:
            alive[state] = any(alive[child] for child in edges[state])
    return layers, edges, alive, traces


def auc(labels, scores):
    # Mann-Whitney rank AUC with average ranks for exact score ties.
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positive = labels.sum()
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    rank_sum = 0.0
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and scores[order[end]] == scores[order[begin]]:
            end += 1
        rank_sum += labels[order[begin:end]].sum() * ((begin+1+end)/2)
        begin = end
    return float((rank_sum-positive*(positive+1)/2)/(positive*negative))


def pruned(problem, edges, values, tau):
    live = {tuple(sorted(problem["numbers"]))}
    expanded = proposed = judged = merged = pruned_count = rounds = 0
    traces = []
    solved = False
    while live:
        rounds += 1
        expanded += len(live)
        nxt = set()
        raw = 0
        for state in live:
            raw += sum(edges[state].values())
            nxt.update(edges[state])
        proposed += raw
        merged += raw - len(nxt)
        if all(len(s) == 1 for s in nxt):
            solved = (problem["target"],) in nxt
            traces.append({"depth": rounds, "raw_candidates": raw, "unique_states": len(nxt), "pruned": 0})
            break
        judged += len(nxt) if tau is not None else 0
        kept = {state for state in nxt if tau is None or values[state] >= tau}
        dropped = len(nxt)-len(kept)
        pruned_count += dropped
        traces.append({"depth": rounds, "raw_candidates": raw, "unique_states": len(nxt), "pruned": dropped})
        live = kept
    return {"solved": solved, "states_expanded": expanded, "states_proposed": proposed,
            "states_judged": judged, "states_merged": merged, "states_pruned": pruned_count,
            "sequential_rounds": rounds, "percentage_pruned": pruned_count/max(judged,1),
            "state_work": expanded+proposed+judged, "depth_trace": traces}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights/countdown_judge.pt"))
    ap.add_argument("--sizes", default="6,7,8")
    ap.add_argument("--counts", default="100,100,20")
    ap.add_argument("--hard-main", type=int, default=250)
    ap.add_argument("--thresholds", default="0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.95,0.99")
    ap.add_argument("--state-cap", type=int, default=500000)
    ap.add_argument("--seconds", type=float, default=120)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    problems = []
    for size, count in zip(map(int,args.sizes.split(",")),map(int,args.counts.split(","))):
        problems += [p for p in manifest["generalisation"] if p["n_numbers"] == size][:count]
    problems += [p for p in manifest["main"] if p["cohort"] == "stratified" and p["stratum"] in ("1","2-5")][:args.hard_main]
    thresholds = list(map(float,args.thresholds.split(",")))
    config = {"manifest_hash": digest(args.manifest), "weights_hash": digest(args.weights),
              "problem_ids": [p["problem_id"] for p in problems], "thresholds": thresholds,
              "state_cap": args.state_cap, "seconds": args.seconds,
              "source_hash": digest(__file__)}
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("changed immutable configuration")
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment())
    done = {r["problem_id"] for r in read_rows(out / "problems.jsonl")}
    judge = load_judge(args.weights)
    t0 = time.perf_counter()
    for index, p in enumerate(problems):
        if p["problem_id"] in done:
            continue
        started = time.perf_counter()
        try:
            layers, edges, alive, compression = graph(p,args.state_cap,args.seconds)
        except (RuntimeError, TimeoutError, MemoryError) as error:
            append(out / "problems.jsonl", {"problem_id": p["problem_id"], "n_numbers": p["n_numbers"],
                   "feasible": False, "error": str(error), "wall_seconds": time.perf_counter()-started})
            print(f"{index+1}/{len(problems)} size {p['n_numbers']}: retained failed attempt: {error}",flush=True)
            continue
        states = sorted(edges)
        values = {}
        score_started = time.perf_counter()
        for begin in range(0,len(states),1024):
            batch = states[begin:begin+1024]
            values.update(zip(batch,judge(batch,p["target"])))
        score_seconds = time.perf_counter()-score_started
        labels = [alive[s] for s in states]
        scores = [values[s] for s in states]
        area = auc(labels,scores)
        # Full state-level outputs regenerate AUC and false-negative curves.
        for state in states:
            append(out / "state_predictions.jsonl", {"problem_id": p["problem_id"],
                   "state": state, "alive": alive[state], "score": values[state]})
        base = pruned(p,edges,values,None)
        paths = layers[-1].get((p["target"],),0)
        cohort = p.get("cohort", "hard" if p["n_numbers"] == 4 else "random")
        for tau in [None]+thresholds:
            begin = time.perf_counter()
            result = pruned(p,edges,values,tau)
            positive = sum(labels)
            false_negative = sum(a and s < tau for a,s in zip(labels,scores)) if tau is not None else 0
            result.update(problem_id=p["problem_id"], n_numbers=p["n_numbers"], cohort=cohort,
                          seed=0, budget=None, threshold=tau, method="no_pruning" if tau is None else "learned_pruning",
                          auc=area, alive_states=positive, false_negative_states=false_negative,
                          false_negative_rate=false_negative/max(positive,1), solution_paths=paths,
                          solve_retention=int(result["solved"])/max(int(base["solved"]),1),
                          expansion_work_reduction=1-result["states_expanded"]/max(base["states_expanded"],1),
                          total_state_work_reduction=1-result["state_work"]/max(base["state_work"],1),
                          wall_seconds=time.perf_counter()-begin, judge_score_seconds=score_seconds,
                          score_mode="frozen_judge_replay", generated_tokens=0,prompt_tokens=0)
            append(out / "raw.jsonl",result)
        append(out / "problems.jsonl", {"problem_id": p["problem_id"],"n_numbers":p["n_numbers"],
               "feasible":True,"cohort":cohort,"solution_paths":paths,"auc":area,
               "states":len(alive),"compression":compression,"graph_and_score_wall_seconds":time.perf_counter()-started})
        print(f"{index+1}/{len(problems)} size {p['n_numbers']}: {len(alive)} exact states, {paths} solution paths, {time.perf_counter()-t0:.0f}s total",flush=True)
    write_json(out / "complete.json", {"attempted_problems":len(problems),"completed_records":len(read_rows(out / "problems.jsonl"))})


if __name__ == "__main__":
    main()
