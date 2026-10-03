"""Regenerate judge-threshold and MBPP tables/figures from problem-level outputs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.submission.analyze import interval,sign_test
from experiments.submission.common import digest,read_rows,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kind",choices=("judge","mbpp"),required=True)
    ap.add_argument("--raw",required=True)
    ap.add_argument("--problems",help="judge feasibility records")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    rows=read_rows(args.raw)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261003)
    groups=defaultdict(list)
    if args.kind=="judge":
        for row in rows:
            cohorts=["all"]
            if row["solution_paths"]<=24:
                cohorts.append("rare_le24_paths")
            for cohort in cohorts:
                groups[(row["n_numbers"],cohort,row["threshold"])].append(row)
        summary=[]
        for (size,cohort,tau),group in sorted(groups.items(),key=lambda v:str(v[0])):
            solved=[int(r["solved"]) for r in group]
            alive=sum(r["alive_states"] for r in group)
            judged=sum(r["states_judged"] for r in group)
            summary.append({"n_numbers":size,"cohort":cohort,"threshold":tau,"problems":len(group),
                            "solve_retention":float(np.mean(solved)),"solve_ci95":interval(solved,rng),
                            "state_pooled_false_negative_rate":sum(r["false_negative_states"] for r in group)/max(alive,1),
                            "state_pooled_percentage_pruned":sum(r["states_pruned"] for r in group)/max(judged,1),
                            "mean_problem_auc":float(np.mean([r["auc"] for r in group if r["auc"] is not None])),
                            "mean_expansion_work_reduction":float(np.mean([r["expansion_work_reduction"] for r in group])),
                            "mean_total_state_work_reduction":float(np.mean([r["total_state_work_reduction"] for r in group]))})
        if args.problems:
            feasibility=read_rows(args.problems)
            write_json(out/"feasibility.json",[{"n_numbers":size,"attempted":sum(r["n_numbers"]==size for r in feasibility),
                       "exact_graph_completed":sum(r["n_numbers"]==size and r["feasible"] for r in feasibility),
                       "failed_attempts":[r for r in feasibility if r["n_numbers"]==size and not r["feasible"]]}
                       for size in sorted({r["n_numbers"] for r in feasibility})])
    else:
        for row in rows:
            groups[(row["method"],row["budget"])].append(row)
        summary=[]
        per_method={}
        for (method,budget),group in sorted(groups.items()):
            by_problem=defaultdict(list)
            for row in group:
                by_problem[row["problem_id"]].append(int(row["solved"]))
            means={p:float(np.mean(v)) for p,v in by_problem.items()}
            per_method[(method,budget)]=means
            summary.append({"method":method,"budget":budget,"problems":len(means),"trials":len(group),
                            "solve_rate":float(np.mean(list(means.values()))),"problem_cluster_ci95":interval(list(means.values()),rng),
                            "generated_tokens_mean":float(np.mean([r["generated_tokens"] for r in group])),
                            "prompt_tokens_mean":float(np.mean([r["prompt_tokens"] for r in group])),
                            "rounds_mean":float(np.mean([r["sequential_rounds"] for r in group]))})
        paired=[]
        for (method,budget),means in per_method.items():
            full=per_method.get(("interference_search",budget),{})
            ids=sorted(set(full)&set(means))
            if not ids or method=="interference_search":
                continue
            diff=[full[p]-means[p] for p in ids]
            paired.append({"comparison":"interference_search minus "+method,"budget":budget,
                           "problems":len(ids),"effect":float(np.mean(diff)),"ci95":interval(diff,rng),
                           "paired_problem_sign_test_p_unadjusted":sign_test(diff)})
        write_json(out/"paired.json",paired)
    write_json(out/"summary.json",summary)
    write_json(out/"receipt.json",{"raw_hash":digest(args.raw),"rows":len(rows),"bootstrap_seed":20261003,
               "scope":"descriptive and exploratory; judge results conditional on fixed seed-0 checkpoint; no multiple-comparison-adjusted significance claim"})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.kind=="judge":
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            for size in sorted({r["n_numbers"] for r in summary}):
                for cohort in ("all","rare_le24_paths"):
                    group=sorted([r for r in summary if r["n_numbers"]==size and r["cohort"]==cohort and r["threshold"] is not None],key=lambda r:r["threshold"])
                    if not group:
                        continue
                    for ax,name in zip(axes,("solve_retention","state_pooled_false_negative_rate","mean_expansion_work_reduction")):
                        ax.plot([r["threshold"] for r in group],[r[name] for r in group],marker=".",label=f"n={size} {cohort}")
                        ax.set(xlabel="Pruning threshold",ylabel=name,ylim=(0,1.02))
            axes[0].legend(fontsize=6)
        else:
            fig,ax=plt.subplots(figsize=(8,5))
            for method in sorted({r["method"] for r in summary}):
                group=sorted([r for r in summary if r["method"]==method],key=lambda r:r["budget"])
                ax.plot([r["budget"] for r in group],[r["solve_rate"] for r in group],marker="o",label=method)
            ax.set(xlabel="Token budget",ylabel="Solve rate",ylim=(0,1.02))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out/"curves.png",dpi=180)
        fig.savefig(out/"curves.pdf")
        plt.close(fig)
    except ImportError:
        pass
    print(f"summarized {len(rows)} rows",flush=True)


if __name__=="__main__":
    main()
