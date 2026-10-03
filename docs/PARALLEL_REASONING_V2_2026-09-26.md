# Interference Search: shared parallel reasoning graph

Status: inference-time research prototype, 26 September 2026. This continues the released paper's level-synchronous reasoning over merged states. It is not parallel token decoding, and it does not establish a new LLM capability or quantum speedup.

## The engine

`interference_search/parallel_reasoning.py` keeps one graph of canonical environment states. A batched proposer is called once for every live level. Equal child states share a node; incoming path weights are pooled with `logaddexp`. A verifier-certified dead state is recorded once, and deadness can propagate backward when all of a parent's legal children are proven dead. Scored states outside the live width remain in a reserve. If the chosen frontier dies, the reasoner can recover an alternative without restarting from the root. Each completed solution retains an executable action witness.

This is a classical graph search structure. The model can serve as proposer or judge, but the graph's merge and proof rules do not depend on model training. The engine counts state expansions, proposal calls and candidates, judged states, merge events, rescue rounds, graph size, and wall time. It keeps level batches at one depth and rejects a merge key reached at different depths, since time or remaining resources would then be missing from the state key.

## Same-judge Countdown comparison

I used the frozen released Countdown judge and the same generated six- and seven-number problem seeds as the paper's frontier comparison. All arms had the same state-expansion budget. The plain shared graph matched the released frontier's solve counts at every tested budget. This is a useful parity check for the new scheduler.

| Problems | Budget | Released frontier | Shared graph + recovery | + path-mass ranking |
|---|---:|---:|---:|---:|
| 100 six-number | 20 | 33 | 35 | 38 |
| 100 six-number | 50 | 61 | 61 | 61 |
| 100 six-number | 100 | 77 | 77 | 73 |
| 100 six-number | 200 | 88 | 88 | 88 |
| 100 six-number | 400 | 94 | 94 | 92 |
| 30 seven-number | 20 | 9 | 10 | 6 |
| 30 seven-number | 50 | 16 | 17 | 14 |
| 30 seven-number | 100 | 21 | 22 | 18 |
| 30 seven-number | 200 | 26 | 26 | 25 |
| 30 seven-number | 400 | 26 | 26 | 26 |

Recovery helps a little at tight budgets and does not improve the high-budget ceiling. Path-mass ranking is inconsistent and often harmful, so it is an ablation rather than a selected feature. At a 100-expansion budget, the six-number released frontier averaged **59.5 ms/problem** versus **68.4 ms** for the recovery graph; at seven numbers it was **91.3 ms** versus **105.7 ms**. The graph currently pays for stored edges, proof propagation, and mass recomputation. The result is an engineering substrate, not a faster search claim. Full receipt: [`parallel_reasoning_v2.json`](../results/countdown/parallel_reasoning_v2.json).

I then replaced repeated whole-graph dead-proof scans with parent-queue propagation and deferred path-mass recomputation when mass is not used to rank. Solve counts stayed identical. At 100 expansions the recovery graph averaged **67.6 ms** on six-number problems and **102.3 ms** on seven-number problems, still slower than the released loop. [Optimized receipt](../results/countdown/parallel_reasoning_v2_optimized.json).

## Model proposal checks

I pinned the non-thinking `Qwen/Qwen3-4B-Instruct-2507` checkpoint at `cdbee75f17c01a7cc42f958dc650907174af0554` and asked it for one indexed arithmetic move per canonical state. In one GPU batch it produced **16/16 valid JSON actions**, but two stochastic samples per state reached only **one distinct successor per state**. The samples repeatedly chose `0 + 1`. This reproduces the paper's warning that state prompting alone need not create useful branch diversity. [Format receipt](../results/model_action_proposer_smoke.json); [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

I then scored every legal structured action under the same frozen model, merged actions reaching the same child state, and evaluated the top children with the exact solver **after** ranking. On 16 alive held-out states, its first child stayed live **2/16** times; one of the top four stayed live **8/16** times. Four unique legal children chosen uniformly would stay live **9.85/16 in expectation** on those states. Pooling action probability across equal child states tied ranking by the best single action in this slice. The model required 32 batched forward calls and 14,833 input tokens to score 150 actions, taking 1.55 seconds. It is not a useful proposer in this test. [Scoring receipt](../results/action_likelihood_alive16.json).

## What follows from this

The paper's merged-frontier result still holds. The shared graph adds recoverability and proof accounting, but the observed solve gain is small and the implementation is slower on Countdown. Prompting or scoring a stock 4B model has not supplied the missing action diversity or value judgment. More stochastic copies of the same prompt would repeat work.

The next meaningful domain must provide **verifiable refutations of reusable subspaces**, so one failure removes paths that were not individually expanded. A finite compositional program-induction grammar can provide that first controlled test, with a deterministic cached solver as the comparator. For open code, matching a few visible tests remains a soft behavior cluster; hard merging requires a sound equivalence criterion. The acceptance test is improved solve rate per total model and verifier work, end-to-end latency, and recovery on held-out structures. A small Countdown gain alone will not substantiate the wider claim.

That controlled test is now recorded in the [counterexample search note](COUNTEREXAMPLE_PARALLEL_REASONING_2026-09-27.md). Certified exclusions and compact recovery work as intended, but the cached factor solver matches solve outcomes, and eight-wide batches spend substantially more verifier calls for a modest reduction in dependent rounds.

The next [async native runtime preflight](ASYNC_NATIVE_EXECUTION_2026-09-27.md) shifts the performance target to concurrent model and execution work, with measured H100 prefix reuse through the full backbone. Its speed receipts are separate from a trained reasoning result.

The later [sandboxed MBPP gate](MBPP_REAL_EXECUTION_GATE_2026-09-27.md) measured that runtime with optimized vLLM generation and real Python tests. It found a modest latency gain at equal visible-test solves, mostly from early sibling cancellation; it did not train or evaluate the native frontier head on code tasks.
