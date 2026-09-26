# One H100 midtraining plan

Status: architecture gate still closed, 27 September 2026. The [compact-state pilot report](COMPACT_STATE_PILOT_2026-09-26.md) records full-checkpoint feasibility and a negative matched training result. The [counterexample search test](COUNTEREXAMPLE_PARALLEL_REASONING_2026-09-27.md) then showed exact compact sharing, but no solve advantage over cached factors and only modest parallel-depth savings at much higher verifier work. The resource estimates below remain a planning reference; they are not a go decision for training either mechanism.

## Decision

Use `Qwen/Qwen3-4B-Base` as the first backbone. It is a text model with 4.0B parameters, a conventional Qwen3 decoder, and an Apache 2.0 model card. The newer Qwen3.5-4B-Base is multimodal and has a more involved text architecture. That may be useful later, but it adds integration work to the first test. Pin the chosen checkpoint to an exact Hugging Face revision before any run.

The target is not to teach a 4B model general language from scratch. The target is to see whether continued training can make a pretrained model propose and judge inside a shared, level-synchronous frontier with verified inhibitory feedback. A 4B result would be evidence about the mechanism at 4B. It would not make a frontier-capable assistant by itself.

The user-provided host at `209.20.157.48` has one H100 PCIe with about 80 GB of VRAM. It was idle during the 26 September preflight, and the project has an isolated Python environment at `/home/ubuntu/interference-search-v2`. The host's billing rate is unknown here. For comparison only, one H100 SXM at Runpod's listed $3.49 per hour costs $83.76 for 24 hours or $167.52 for 48 hours in GPU rental. That rate does not price this PCIe host. The time range is a planning envelope, not a measured training schedule or approved spend cap.

## Training configuration to preflight

The architecture arm adds state encoders and action heads around the tested `NativeFrontierCell`, then connects its task and state vectors to selected decoder layers. It trains the new module and all text-decoder weights if the full-model memory smoke fits. Use BF16, activation checkpointing, FlashAttention or SDPA, and a memory saving optimizer. A normal mixed precision AdamW run needs roughly 18 bytes per parameter before activations: 4B parameters alone consume about 72 GB. The optimizer choice must be fixed before comparison and logged. A fallback that freezes most of the decoder changes the claim to architecture adaptation, and results must be named that way.

Unsloth is an option for this run. Its documentation supports continued pretraining, Qwen fine-tuning, and full fine-tuning for supported stock models. The fast kernels and training wrappers have not been verified for this new recurrent path, and Unsloth is not installed in the isolated GPU environment. The plain PyTorch tiny-model path now passes on CPU and H100. If using Unsloth, load identical weights into both paths and compare forward outputs, loss, gradients on every new module, one optimizer update, and save/reload behavior. Repeat on the full checkpoint before committing the training clock. If Unsloth skips the new path, changes its gradients, or cannot restore the combined artifact, use the reference trainer for both arms. Record achieved tokens per second and peak VRAM instead of importing Unsloth's published speedup claim. A LoRA continued-pretraining run must be named adapter training; use Unsloth's full-finetuning path only if it updates the intended full set of weights.

For the matched control, use the same pinned backbone, training records, update count, optimizer, trainable parameter count as closely as possible, and measured GPU time. Compare a single-state scratchpad, the shared frontier cell with zero refutation coupling, and the cell with inhibitory coupling. If a control gets fewer labels, less compute, or fewer trainable parameters, the comparison cannot isolate the shared state or inhibitory path.

Keep the verifier outside the weights. It executes model-proposed candidates and returns exact feedback. The model must propose candidate operations and score states in the shared frontier. Exact keys merge true repeats; matching outcomes on a finite test list are soft clusters. A hard exclusion may cover a subspace only when the verifier provides a sound certificate for that whole subspace; otherwise it excludes the tested candidate only. Count verifier calls and time in every arm.

## Data and evaluation gate

Before renting a GPU, freeze a Countdown bootstrap curriculum and a generated program-induction curriculum with source-disjoint validation and test sets. The Countdown records must include model-selected action labels, executed successor states, reachability labels, and paired states before and after one verified refutation. Those pairs train the successor predictor and expose whether the negative path changes a sibling's action. The second family should have multiple possible programs consistent with early evidence and counterexamples that distinguish them. Keep the released cached frontier, an exact solver, and a cached state enumerator, with their full runtime, as baselines. The current register-program family is solved cheaply by cached factors, so it fails this gate as a primary midtraining target. A replacement must first show that learned proposals or shared-state updates add value on held-out instances beyond the cached solver.

Measure at least these outcomes on held-out instances:

- Exact solve rate against model tokens, total FLOPs, verifier calls, and wall-clock time.
- Peak memory and factor-state size as candidate path count grows.
- Verified suppression of unvisited invalid candidates, false deletion of valid candidates, and recovery after the top candidate is refuted.
- General language and code regression against the exact parent checkpoint, since the 4B backbone is being changed.
- Blank, shuffled, and disabled factor-state ablations. The mechanism claim needs a behavioral drop when the correct state path is removed.

Use paired examples and repeated seeds for any claimed win. A single 24 to 48 hour run will probably be a feasibility result, not a decisive statistical result.

## Clock budget and stop conditions

The [local tiny-model smoke](../results/native_lm_smoke.json) now completes forward, backward, an optimizer update that changes backbone and frontier weights, and checkpoint save/reload with equal outputs. It used random inputs, so held-out task behavior is still untested. Before renting, build and freeze the task data and eval harness. The full-model H100 smoke then records actual peak VRAM, tokens per second, optimizer step time, and checkpoint time. Use those measurements to decide the token and update budget for each arm. Do not infer it from theoretical H100 FLOPs.

For a 48-hour ceiling, reserve time for both arms, evaluation, and failures instead of spending the entire window on one model. A provisional allocation is 4 hours for setup and smokes, 16 hours per training arm, 6 hours for frozen evaluation and ablations, and 6 hours of contingency. Shorter ceilings scale down the training arms together. If the full-model setup spills memory, switch both arms to the same selective-weight recipe before training. If the architecture arm costs more per update, compare at matched total GPU time as well as matched tokens.

Stop a run that cannot save/reload, cannot pass a basic exact-verifier check, leaks test examples into training, or cannot produce a fair control within the available clock. Save the failure log and report it as a preflight result.

## Sources

- [Qwen3-4B-Base model card](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Runpod GPU pricing](https://www.runpod.io/pricing)
- [Hugging Face model memory anatomy](https://huggingface.co/docs/transformers/v4.32.0/model_memory_anatomy)
- [Unsloth continued pretraining](https://unsloth.ai/docs/basics/continued-pretraining)
- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
