# Interference Search submission experiments

This package is the reproducible experimental harness for the expanded Countdown and MBPP evaluations. The compute runs use `spark-d580`; the repository contains the scripts and frozen input definitions, while the large raw receipts are kept under `~/interference-search-submission-20261003` on that machine.

## Countdown

`experiments/submission/build_countdown.py` creates a frozen source-disjoint manifest. The current Spark manifest is `data/submission/countdown-v1.json`: 1,000 four-number problems, split into 500 random problems and 125 problems in each solution-path stratum (`1`, `2-5`, `6-20`, `>20`). It also contains 100 random problems for each of six-, seven-, and eight-number transfer tests. `build_hard_six.py` creates a separate 100-problem six-number set with 1–24 exact solution paths for the rare-solution threshold curve.

The exact path count is the multiplicity of legal operation sequences produced by `moves()`. Commutative operations are counted once per unordered indexed pair; equal input values retain their index multiplicity. The manifest is frozen before judge scores or thresholds are inspected.

`experiments/submission/countdown_engine.py` implements the factorial switches:

* state merging on or off;
* learned judge/pruning on or off;
* level-synchronous advancement on or off.

It also runs single-line search, BFS, beam, best-first, independent parallel search, frontier without merging, merging without the learned judge, and full Interference Search. Every run charges state work against a hard budget and records proposed, merged, judged, pruned, expanded, rounds, terminal failures, frontier drops, wall time, and a depth-level raw-to-unique state trace.

`experiments/submission/run_countdown.py` writes one JSONL row per problem × method × seed × budget. The main Spark run uses 1,000 problems, 17 methods, ten search seeds, and budgets 25, 50, 100, 200, 500, and 1,500. The width-4 and proposal-count runs are separate immutable controls. `experiments/submission/analyze.py` regenerates summary tables, problem-cluster bootstrap intervals, paired contrasts, factorial effects, compression tables, and budget curves from raw JSONL only.

`experiments/submission/judge_transfer.py` builds exact state graphs and evaluates the learned judge at thresholds 0, .05, .1, .2, .3, .5, .7, .9, .95, and .99. It saves AUC, false-negative rate, solve retention, percentage pruned, expansion reduction, total state-work reduction, and the exact graph compression. Graphs that exceed the configured state/time cap are recorded as failed feasibility attempts. `train_judges.py` retrains five independent judge seeds on the original four- and five-number sources; `replay_judge_seeds.py` applies those checkpoints to the identical frozen transfer graphs.

## MBPP

`experiments/submission/run_mbpp.py` runs the pinned `mlx-community/Qwen3-1.7B-4bit` checkpoint on Spark through a Transformers SDPA backend. `spark_qwen.py` records the source revision, checkpoint hash, affine expansion, dtype, backend, device, parameter count, and load time. The Linux backend has not been declared MLX-kernel equivalent; all MBPP comparisons are within this one backend.

The MBPP manifest is a fixed SHA-256 ordering of 200 sanitized problems. The current suite has five seeds, budgets 1,500 and 4,000 total tokens, and six arms: independent sampling, transcript, linear state, frontier without merging, Interference Search, and legacy behavior merging. Prompts, completion token IDs, generated text, prompt tokens, execution results, candidate code, merge keys, round traces, and per-problem final rows are retained.

Generated code is executed by `spark_sandbox.py`. Spark does not permit the user namespace setup required by bubblewrap, so the runner uses unprivileged Landlock filesystem rules, seccomp syscall denial, resource limits, and a preflight that probes file reads/writes, sockets, and process creation. A sandbox setup failure is an infrastructure failure and stops admission; it is not silently counted as a model error.

## Reproduction commands on Spark

From the checkout on Spark:

```bash
python -m experiments.submission.build_countdown \
  --out data/submission/countdown-v1.json --total 1000 --transfer 100

python -m experiments.submission.run_countdown \
  --manifest data/submission/countdown-v1.json \
  --out results/submission/main

python -m experiments.submission.analyze \
  --raw results/submission/main/raw.jsonl \
  --out results/submission/main/analysis

python -m experiments.submission.judge_transfer \
  --manifest data/submission/countdown-v1.json \
  --out results/submission/judge-transfer

python -m experiments.submission.summarize_auxiliary \
  --kind judge --raw results/submission/judge-transfer/raw.jsonl \
  --problems results/submission/judge-transfer/problems.jsonl \
  --out results/submission/judge-transfer-summary

python -m experiments.submission.run_mbpp \
  --data data/mbpp_sanitized.json \
  --model models/qwen3-1.7b-mlx4bit \
  --out results/submission/mbpp-main \
  --n 200 --seeds 0,1,2,3,4 --budgets 1500,4000
```

`complete.json`, `config.json`, `environment.json`, and the source hashes in each output directory are admission receipts. A complete row count is required before a table or figure is reported. `archive_negative.py` preserves prior ledger, rewind, cross-stream, prompting, toy, interrupted, and MBPP partial results with hashes; the historical 9/30 versus 8/30 MBPP aggregate has no recoverable paired rows and is not converted into a significance claim.

## Current evidence boundary

The first 1,000-problem factorial is complete. At width 16 on the random cohort, best-first solves 72.3%, 93.3%, 100%, and 100% at budgets 100, 200, 500, and 1,500. Full Interference Search solves 0%, 4.3%, 98.8%, and 98.8%. The three-way factorial interaction is −5.2, −0.8, +2.2, and +2.2 percentage points at those budgets; the 95% problem-cluster intervals at 200 include zero, while the 500 and 1,500 intervals are positive. On the rare `>20` stratum, the full method reaches 100% at 500 while single-line search reaches 98.5%, but best-first also reaches 100%.

These results establish a budget- and width-dependent interaction question. They do not support a claim that Interference Search dominates existing search strategies. The final report should lead with the complete curves, the matched budgets, the component effects, the hard-problem judge threshold curve, and the negative historical experiments.

The exact transfer graph completed for all 100 six-number problems, 79 of 100 seven-number problems, and none of the eight-number problems under the 500,000-state cap. Every rejected graph is retained with its state-cap failure and wall time. The eight-number manifest therefore remains a frozen attempted transfer set, not a measured eight-number result.
