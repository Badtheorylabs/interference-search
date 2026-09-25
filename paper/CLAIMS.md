# Claims and evidence

Every number in the paper, where it comes from, and how to reproduce it. Paths are relative to the repository root.

| Claim | Value | Evidence | Reproduce |
|---|---|---|---|
| Hard 4-number Countdown, 200 judged positions: Interference Search vs one line, same judge | 30/30 vs 21/30 | `results/countdown/countdown_benchmark.json` | `cd experiments/countdown && python benchmark.py` |
| Sequential steps to solve all 30 (1,500 judged positions) | 3.0 vs 23.7 | same benchmark, rounds on solved problems | as above with `--budgets 150,200,500,1500` |
| Paths per state at six numbers | 831,176 paths, 13,229 states, 62.8× | `results/countdown/compression.txt` | `python compression.py` |
| Share of complete paths through a dead state after two moves | 95.4%, 96.2%, 82.6% (4, 5, 6 numbers) | `results/countdown/compression.txt` | same |
| Judge on unseen random 6-number problems | 100% solved at 9.3% of the search (τ = 0.7) | `results/frontier/refuter_results_s0.json` | `python train_judge.py` |
| Judge on hard 6-number problems (≤ 24 solution paths) | 98% at 17.9% (τ = 0.2); 43% at 4.3% (τ = 0.7) | training-run log in `docs/RESEARCH_LOG.md` | `python train_judge.py` |
| Judge on 7-number problems (10) | 100% at 7.9% (τ = 0.7) | same | same |
| Unseen 6-number, 100 expansions: frontier, line, frontier without merge, best-first | 77%, 40%, 50%, 10% | `results/countdown/frontier_vs_linear.json` | `python frontier_vs_linear.py --ablation` |
| Qwen3-1.7B thinking in text on the 30 hard problems | 3/30 | `experiments/llm/llm_state.py`, transcript arm | `python llm_state.py --arms transcript` |
| Judges inside the frontier (prompted, hidden-state probe, feature probe, trained) | 4, 15, 22, 23 of 30; AUC 0.58, 0.87, 0.94, 0.98 | `results/llm/probe_results.json` (prompted, hidden-state probe, trained judge); `probe_control.py` output (feature probe) | `cd experiments/llm && python probe.py && python probe_control.py` |
| Correct expression written at token 1,313, no final answer | 17 attempts, 11 repeats | `results/llm/cot_trace.json` | `python cot_trace.py` |
| Failed attempts repeating an own or sibling refutation (Phase 0) | 45.1% and 29.7% | `results/llm/phase0_qwen3-1.7b.jsonl` | `python phase0.py && python analyze0.py …` |
| Textual notes prime retries (Phase 1) | 33.6% vs about 21% within 80 tokens | `results/llm/phase1_qwen3-1.7b.jsonl` | `python phase1.py && python analyze1.py …` |
| Code, 30 MBPP first-try failures | 7, 7, 8, 9 of 30; 83% merged | `results/code/bench_code_partial.json` | `cd experiments/code && python benchmark.py` |
| Chain of thought, generated tokens per solved Countdown problem | about 14,900 (44,703 over 3 solves) | `results/llm` transcript arm, 1,490 generated per problem | `python llm_state.py --arms transcript` |
| Judge arms: tokens read per problem, generated | about 4,290 read (65 states × 66 tokens), 0 generated | probe run judged-state counts; prompt length measured with the Qwen3 tokenizer | `python probe.py` |
| Wall clock, one problem at a time on an M2: Qwen3-1.7B thinking in text vs Qwen3-1.7B hidden-state probe judging inside the search | 57.4 s and 1/30 vs 15.2 s and 15/30 | `results/llm/cot_timing.json`, `results/llm/probe_search_timing.json` | `cd experiments/llm && python cot_timing.py && python probe.py --timing-only` |
| Code, generated tokens per solve | 4,560 (Interference Search), 5,090 (best of N), 5,460 (revise latest) | `results/code/bench_code_partial.json` | `cd experiments/code && python benchmark.py` |
| Parallel streams trained from scratch, isolated vs visible (4 streams) | 0.508 vs 0.511 | research log, toy section | `experiments/toy/run_gpu.sh` |

All runs are one seed. Results on 30 problems carry roughly ±2.5 problems of sampling noise at these solve rates.
