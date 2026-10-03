"""Regenerate tables, cluster intervals, factorial contrasts, and figures from raw rows."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.submission.common import digest, write_json


def interval(values, rng, replicates=2000):
    values = np.asarray(values, dtype=float)
    samples = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)])
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def sign_test(values):
    wins, losses = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = wins + losses
    if not n:
        return 1.0
    # stable exact two-sided binomial tail, per problem (not per seed row)
    tail = sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2))
               for k in range(min(wins, losses)+1))
    return min(1.0, 2*tail)


def analyze(raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261003)
    grouped = defaultdict(lambda: defaultdict(list))
    metrics = defaultdict(lambda: defaultdict(list))
    compression = defaultdict(lambda: [0, 0, 0])
    seen = set()
    for line in Path(raw).open():
        row = json.loads(line)
        key = (row["problem_id"], row["method"], row["seed"], row["budget"])
        if key in seen:
            raise ValueError(f"duplicate trial {key}")
        seen.add(key)
        for cohort in ("all", row["cohort"], "stratum_" + row["stratum"]):
            group = (cohort, row["method"], row["budget"])
            grouped[group][row["problem_id"]].append(int(row["solved"]))
            for name in ("states_proposed", "states_judged", "states_expanded", "states_merged",
                         "states_pruned", "sequential_rounds", "wall_seconds", "work_used"):
                metrics[group][name].append(row[name])
        for depth in row["depth_trace"]:
            group = (row["method"], row["budget"], depth["depth"])
            compression[group][0] += depth["raw_candidates"]
            compression[group][1] += depth["unique_states"]
            compression[group][2] += 1
    summary = []
    for (cohort, method, budget), problems in sorted(grouped.items()):
        values = [np.mean(v) for v in problems.values()]
        summary.append({"cohort": cohort, "method": method, "budget": budget,
                        "problems": len(values), "trials": sum(map(len, problems.values())),
                        "solve_rate": float(np.mean(values)),
                        "problem_cluster_ci95": interval(values, rng),
                        **{name + "_mean": float(np.mean(v)) for name, v in metrics[(cohort, method, budget)].items()}})
    contrasts = []
    budgets = sorted({g[2] for g in grouped})
    for cohort in sorted({g[0] for g in grouped}):
        for budget in budgets:
            cells = {f"{m}{j}{s}": grouped.get((cohort, f"factorial_m{m}j{j}s{s}", budget), {})
                     for m in (0,1) for j in (0,1) for s in (0,1)}
            ids = set.intersection(*(set(v) for v in cells.values()))
            if ids:
                for label, coefficients in (
                    ("three_way_interaction", {k: (-1)**(3-sum(map(int,k))) for k in cells}),
                    ("merging_main_effect", {k: (1 if k[0]=="1" else -1)/4 for k in cells}),
                    ("judge_main_effect", {k: (1 if k[1]=="1" else -1)/4 for k in cells}),
                    ("synchrony_main_effect", {k: (1 if k[2]=="1" else -1)/4 for k in cells}),
                    ("merge_judge_interaction", {k: (1 if k[0]==k[1] else -1)/2 for k in cells}),
                    ("merge_sync_interaction", {k: (1 if k[0]==k[2] else -1)/2 for k in cells}),
                    ("judge_sync_interaction", {k: (1 if k[1]==k[2] else -1)/2 for k in cells}),
                    ("full_beyond_sum_of_individual_components", {"111":1,"100":-1,"010":-1,"001":-1,"000":2}),
                    ("merging_effect_with_judge_and_sync", {"111": 1, "011": -1}),
                    ("judge_effect_with_merge_and_sync", {"111": 1, "101": -1}),
                    ("sync_effect_with_merge_and_judge", {"111": 1, "110": -1}),
                    ("full_minus_no_components", {"111": 1, "000": -1}),
                ):
                    values = [sum(c*np.mean(cells[k][p]) for k,c in coefficients.items()) for p in sorted(ids)]
                    contrasts.append({"cohort": cohort, "budget": budget, "contrast": label,
                                      "problems": len(ids), "effect": float(np.mean(values)),
                                      "problem_cluster_ci95": interval(values, rng)})
            full = grouped.get((cohort, "full_interference_search", budget), {})
            for method in sorted({g[1] for g in grouped if not g[1].startswith("factorial_")}):
                if method == "full_interference_search":
                    continue
                other = grouped.get((cohort, method, budget), {})
                shared = sorted(set(full) & set(other))
                if not shared:
                    continue
                values = [np.mean(full[p])-np.mean(other[p]) for p in shared]
                contrasts.append({"cohort": cohort, "budget": budget,
                                  "contrast": "full_minus_" + method, "problems": len(shared),
                                  "effect": float(np.mean(values)), "problem_cluster_ci95": interval(values, rng),
                                  "paired_problem_sign_test_p_unadjusted": sign_test(values)})
    write_json(out / "summary.json", summary)
    write_json(out / "contrasts.json", contrasts)
    write_json(out / "compression.json", [{"method": m, "budget": b, "depth": d,
               "raw_candidates_mean": v[0]/v[2], "unique_states_mean": v[1]/v[2],
               "raw_to_unique_ratio": v[0]/max(v[1],1)} for (m,b,d),v in sorted(compression.items())])
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(out / "analysis_receipt.json", {"raw_sha256": digest(raw), "rows": len(seen),
               "bootstrap_seed": 20261003, "bootstrap_replicates": 2000,
               "interval_scope": "resample source problems after averaging 10 search seeds; conditional on fixed trained judge",
               "multiplicity": "pairwise p-values unadjusted and exploratory; primary prespecified contrast is three-way interaction on random cohort at budget 200",
               "timing_warning": "frozen_judge_replay wall time excludes scoring precompute; use online_uncached runs for end-to-end comparisons"})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for cohort in ("random", "stratified"):
            fig, ax = plt.subplots(figsize=(8,5))
            for method in sorted({r["method"] for r in summary if not r["method"].startswith("factorial_")}):
                rows = [r for r in summary if r["cohort"] == cohort and r["method"] == method]
                if rows:
                    ax.plot([r["budget"] for r in rows], [r["solve_rate"] for r in rows], marker=".", label=method)
            ax.set(xscale="log", xlabel="Charged state-work budget", ylabel="Solve rate", ylim=(0,1.02), title=cohort)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / f"budget_{cohort}.png", dpi=180)
            fig.savefig(out / f"budget_{cohort}.pdf")
            plt.close(fig)
        fig, axes = plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
        for ax, (m,j,s) in zip(axes.flat, ((m,j,s) for m in (0,1) for j in (0,1) for s in (0,1))):
            rows = [r for r in summary if r["cohort"] == "random" and r["method"] == f"factorial_m{m}j{j}s{s}"]
            ax.plot([r["budget"] for r in rows], [r["solve_rate"] for r in rows], marker="o")
            ax.set(xscale="log",title=f"merge={m}, judge={j}, synchronous={s}",ylim=(0,1.02))
        fig.supxlabel("Charged state-work budget")
        fig.supylabel("Solve rate")
        fig.tight_layout()
        fig.savefig(out / "factorial.png",dpi=180)
        fig.savefig(out / "factorial.pdf")
        plt.close(fig)
    except ImportError:
        write_json(out / "plot_unavailable.json", {"reason": "matplotlib not installed; numeric tables complete"})
    print(f"analyzed {len(seen)} trials", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    analyze(args.raw, args.out)
