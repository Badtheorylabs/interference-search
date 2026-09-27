# Native online Interference Search gate

Date: 27 September 2026

## What this gate fixes

The earlier SWE experiment trained a native scorer on static frontier snapshots. That was useful component work, but it did not run the defining Interference Search loop. This gate restores the original synchronized algorithm:

1. the complete live frontier proposes actions through one Qwen3-4B model;
2. the environment executes every selected action;
3. exact convergent states merge and pool their proposal evidence;
4. discarded states enter the learned soft-refutation memory;
5. all survivors advance together to the next round.

The task prompt is encoded once. State branches reuse its KV cache and are encoded by the same model weights. The native head reads model hidden states for survival, action proposal, successor prediction, and negative coupling. Countdown arithmetic execution and state equivalence remain exact.

## Frozen evaluation

The checkpoint is the previously saved 2,000-step Unsloth Qwen3-4B adapter and native frontier head. No training or setting selection used these outcomes. Evaluation used all 200 rows of the five-number structural holdout from `data/native_countdown/test_5_numbers.jsonl`.

Fixed settings:

- frontier width: 12
- state expansion budget: 40
- actions per state: 8
- synchronized rounds: at most 4 for a five-number problem
- same checkpoint, action vectors, prompts, executor, and problems in every arm

| Arm | Solved | Rate | Mean executed actions | Mean exact merges |
|---|---:|---:|---:|---:|
| merge + learned inhibition | 54/200 | 27.0% | 199.6 | 34.0 |
| merge + coupling disabled | 45/200 | 22.5% | 199.1 | 34.4 |
| learned inhibition + merge disabled | 33/200 | 16.5% | 205.3 | 0 |

The full system gained 22 problems and lost 1 against its no-merge ablation. An exact paired sign test gives `p = 5.72e-6`. Exact merging therefore has a strong causal effect in this online native loop: **+10.5 percentage points**, or **1.64x** the no-merge solve rate, at slightly less execution work.

Learned negative coupling gained 15 problems and lost 6 against the same merged engine with coupling set to zero: **+4.5 percentage points**, or **1.20x** the zero-coupling solve rate. The exact paired sign test is `p = 0.0784`. This is a positive signal, not yet conclusive at the conventional 0.05 threshold.

The three arms completed in 331.0 seconds after model setup. Per-problem search latency averaged 0.522 seconds for the full system, 0.563 seconds with coupling disabled, and 0.544 seconds without merging. The latency difference should be treated as descriptive because the arms ran sequentially and the first arm included GPU warmup.

## Claim boundary

This result supports the architecture's strongest established mechanism in a real native-model loop: synchronized parallel execution plus exact convergent-state merging materially improves search, and the learned inhibitory path has a smaller favorable causal signal.

It does not yet establish broad reasoning superiority. The holdout is instance-disjoint and structurally larger than training, but it shares the Countdown generator. The model proposes actions while the environment executes a typed arithmetic action space. There is no matched Qwen text-reasoning baseline in this gate. The checkpoint was trained on supervised frontier snapshots, so it has not yet been optimized end to end on online rollouts.

The next training run should learn from trajectories emitted by this synchronized engine. Its primary admission test is the paired online coupling effect on held-out problems; static snapshot accuracy is secondary.

## Receipts

- source commit: `34d82ecbd64ef1f44c45640c7180406b01e6b63b`
- report: `results/native_online_200/report.json`
- report SHA-256: `92884489d6fc27f151fe6111f7a76043b0b739b93f4a26a259991b1da3c95e61`
- native head SHA-256: `04724d09786a16be8628608daf495b39058b2b70d85b0fb2f42420202bc43a0f`
- action vectors SHA-256: `5cbf4c2079abbc34c6c4e5b510aca7c749f4e2d0dddacd759525f0db7352bf74`
- Unsloth adapter SHA-256: `87537064c58a40f934b7c0578449a3fddc2aa84e8b51b5644d2fd1e9901aecfb`
- GPU: NVIDIA H100 PCIe 80GB
- H100 state after evaluation: no active compute process

