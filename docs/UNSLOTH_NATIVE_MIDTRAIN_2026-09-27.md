# Native Interference Search midtraining on Qwen3-4B

Status: completed 2,000-update mechanism run, 27 September 2026. The run trained a native Interference Search head and rank-32 Unsloth LoRA adapters through Qwen3-4B's live task KV. It learned a causal interference pathway, but the final policy did not beat the untrained starting policy overall. It is a mechanism checkpoint, not a promoted model.

## What was trained

The parent was `Qwen/Qwen3-4B-Base` at revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Unsloth 2026.9.11 injected rank-32 rank-stabilized LoRA into the query, key, value, output, gate, up, and down projections across all 36 layers. That produced 66,060,288 trainable backbone parameters. The native frontier added 85,655,042 trainable parameters for frontier attention, action ranking, survival, successor prediction, and verified-failure matching.

This was native to the model computation. Each task was prefetched once through Qwen. Frontier states, verified dead states, and executed successors branched from the task KV and passed through the same decoder weights. The search head consumed those hidden states directly. Unsloth's optimized inference forward exposed KV but could not backpropagate through this branch pattern because of an in-place operation. I retained Unsloth's injected LoRA modules and adapter saving, then used the reference Qwen3 forward for differentiable KV semantics. The exact forward path is named `unsloth_lora_with_reference_qwen3_kv` in the receipt.

The first 100 steps warmed the native head. LoRA then trained at `1e-5`; the native head used `5e-5`. Interference was disabled through step 1,200 and enabled for the final 800 updates. The action objective assigned probability to every verifier-confirmed winning action instead of one arbitrary canonical action. The transition target was the stop-gradient Qwen hidden representation of the executed successor. The run used 2,000 generated training rows, 100 instance-disjoint four-number validation rows, and 100 five-number structural holdout rows.

## Result

The full run took **1,008.7 seconds**, with a median steady step of **0.458 seconds**. Peak allocated GPU memory was **18.75 GiB**. Loss moved from **3.677** on the first update to **2.254** on the last; the final 25-update mean was **2.644**. Both sampled LoRA tensors and 17 native-head tensors changed. The adapter, head, and fixed 4B-derived action vectors were saved and freshly reloaded with equal measured behavior.

| Split and condition | Winning action top-1 | Known refuted action top-1 | Failure-match gap |
|---|---:|---:|---:|
| Validation, before training | 30.5% | 12.4% | +0.0002 |
| Validation, final weights, interference off | 15.0% | 14.3% | +0.2177 |
| Validation, final weights, interference on | 24.5% | 10.5% | +0.2177 |
| Five-number holdout, before training | 23.9% | 4.8% | approximately 0 |
| Five-number holdout, final weights, interference off | 16.0% | 8.2% | +0.1690 |
| Five-number holdout, final weights, interference on | 19.8% | 6.8% | +0.1690 |

The final weights provide causal evidence for the native pathway. Enabling interference increased winning-action selection by **9.5 percentage points** on validation and **3.8 points** on the five-number holdout compared with the same final weights and coupling disabled. It also reduced selection of known-refuted actions. The failure matcher moved from no separation to a clear positive gap on both splits.

The trained policy still remained below its untrained starting point by **5.9 points** on validation and **4.1 points** on the five-number holdout. The run therefore passed the mechanism gate and failed the capability-promotion gate. Repeating these 2,000 rows for tens of hours would likely deepen the policy regression.

## What should change before a full run

The next run needs substantially more distinct verifier trajectories, checkpoint evaluation during training, and explicit protection for the parent policy. A useful first full curriculum is at least 100,000 distinct trajectories across Countdown sizes, compositional program induction, and sandboxed code repair. The training mix should include parent-policy distillation or a language/code auxiliary objective, multi-step recovery trajectories, and balanced action coverage. Checkpoints should be admitted by held-out search behavior rather than training loss.

At the measured rate, 150,000 updates would take about 19 hours of optimizer time and roughly 22 to 24 hours including periodic evaluation and artifact work. That run should begin only after the expanded curriculum and checkpoint gates are frozen.

## Reproduction and artifacts

The exact training source was commit `6c36063f51a9c26ee4dd54fe6b7c29ba3cfd5827`, with script SHA-256 `3021abfecfafaa5a4739742c7efd5f301556dcc7f5b35210a32cd453e5f8c33e`. Dataset hashes are recorded in the [run report](../results/qwen4b_unsloth_native_midtrain_2k_report.json). The [training log](../results/qwen4b_unsloth_native_midtrain_2k_log.txt) contains every 25-step measurement. Model artifacts remain on the H100 host; their paths, byte sizes, and hashes are in the [artifact manifest](../results/qwen4b_unsloth_native_midtrain_2k_artifacts.json). The H100 was idle after the run.
