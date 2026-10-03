"""Evaluate retrained judges on the identical, already exactly labelled test graphs."""
import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import torch

from interference_search.countdown import moves
from interference_search.judge import load_judge
from experiments.submission.common import append,digest,read_rows,write_json
from experiments.submission.judge_transfer import auc,pruned


class LazyEdges(dict):
    def __missing__(self,key):
        self[key]=Counter(moves(key))
        return self[key]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--canonical",required=True)
    ap.add_argument("--weights",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text())
    problems={p["problem_id"]:p for p in manifest["generalisation"]+manifest["main"]}
    canonical=Path(args.canonical)
    config={"manifest_hash":digest(args.manifest),"canonical_predictions_hash":digest(canonical/"state_predictions.jsonl"),
            "weights_hash":digest(args.weights),"judge_seed":args.seed,"thresholds":[0,.05,.1,.2,.3,.5,.7,.9,.95,.99],
            "source_hash":digest(__file__)}
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=config:
        raise ValueError("changed seed replay config")
    write_json(out/"config.json",config)
    canonical_problems={r["problem_id"]:r for r in read_rows(canonical/"problems.jsonl")}
    write_json(out/"feasibility.json",list(canonical_problems.values()))
    done={r["problem_id"] for r in read_rows(out/"problems.jsonl")}
    judge=load_judge(args.weights)
    start=time.perf_counter()
    with (canonical/"state_predictions.jsonl").open() as stream:
        parsed=(json.loads(line) for line in stream)
        for ident,group in itertools.groupby(parsed,key=lambda r:r["problem_id"]):
            if ident in done:
                for _ in group:
                    pass
                continue
            rows=list(group)
            p=problems[ident]
            states=[tuple(r["state"]) for r in rows]
            labels=[r["alive"] for r in rows]
            values={}
            for begin in range(0,len(states),1024):
                batch=states[begin:begin+1024]
                values.update(zip(batch,judge(batch,p["target"])))
            scores=[values[s] for s in states]
            area=auc(labels,scores)
            edges=LazyEdges()
            base=pruned(p,edges,values,None)
            with (out/"state_predictions.jsonl").open("a") as predictions:
                for state,label,score in zip(states,labels,scores):
                    predictions.write(json.dumps({"problem_id":ident,"state":state,"alive":label,"score":score},separators=(",",":"))+"\n")
            state_sizes=[]
            for size in sorted({len(s) for s in states}):
                sub=[i for i,s in enumerate(states) if len(s)==size]
                state_sizes.append({"remaining_numbers":size,"states":len(sub),"positive":sum(labels[i] for i in sub),
                                    "auc":auc([labels[i] for i in sub],[scores[i] for i in sub])})
            for tau in [None]+config["thresholds"]:
                result=pruned(p,edges,values,tau)
                positive=sum(labels)
                fn=sum(a and score<tau for a,score in zip(labels,scores)) if tau is not None else 0
                result.update(problem_id=ident,n_numbers=p["n_numbers"],cohort=canonical_problems[ident]["cohort"],
                              seed=args.seed,judge_seed=args.seed,budget=None,threshold=tau,method="no_pruning" if tau is None else "learned_pruning",
                              auc=area,alive_states=positive,false_negative_states=fn,false_negative_rate=fn/max(positive,1),
                              solution_paths=canonical_problems[ident]["solution_paths"],solve_retention=int(result["solved"]),
                              expansion_work_reduction=1-result["states_expanded"]/max(base["states_expanded"],1),
                              total_state_work_reduction=1-result["state_work"]/max(base["state_work"],1),score_mode="frozen_judge_replay",
                              generated_tokens=0,prompt_tokens=0,model_hash=config["weights_hash"])
                append(out/"raw.jsonl",result)
            append(out/"problems.jsonl",{"problem_id":ident,"n_numbers":p["n_numbers"],"feasible":True,
                   "judge_seed":args.seed,"auc":area,"state_size_metrics":state_sizes})
            print(f"judge seed {args.seed} replay {ident}, {len(states)} states; {time.perf_counter()-start:.0f}s",flush=True)
    write_json(out/"complete.json",{"judge_seed":args.seed,"completed_problems":len(read_rows(out/"problems.jsonl")),
               "failed_exact_graphs":sum(not r["feasible"] for r in canonical_problems.values())})


if __name__=="__main__":
    main()
