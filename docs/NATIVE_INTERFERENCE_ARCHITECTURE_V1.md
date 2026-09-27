# Native Interference Architecture V1

Status: frozen engineering target, 27 September 2026

## Objective

Build the strongest practical reasoning architecture suggested by the Interference Search results. The released explicit frontier is evidence for execution, convergence, and refutation. It is not a constraint that the native model must reproduce the original controller literally.

The target combines cheap latent parallelism between actions with hard verifier-grounded state at tool boundaries. The checkpoint owns reasoning, action selection, merging, suppression, and collapse. The runtime only executes tool calls and returns results plus exact canonical keys when the domain can provide them.

## Native state

Each sequence carries a persistent register alongside its ordinary KV cache:

- `H [batch, slots, dim]`: compact hypothesis states;
- `alive [batch, slots]`: learned occupancy and survival gates;
- `provenance`: which action and observation last changed each slot;
- `hard_key`: optional environment-certified state identity;
- `confidence`: calibrated slot value and collapse probability;
- `refutations`: verified failure vectors with scope and source.

The first 4B implementation uses eight 128-dimensional slots. Slot count is a maximum capacity, not a requirement to spend all eight on every problem.

## Transformer update

Selected transformer blocks perform four operations inside the normal forward pass:

1. **Write:** the current token/tool observation updates relevant hypothesis slots.
2. **Constructive interaction:** compatible slots exchange support and can converge.
3. **Signed refutation:** verified failure vectors subtract from matching slots and action regions.
4. **Read:** the active register conditions the next token, tool action, or collapse decision.

Unverified model guesses may change soft scores but cannot create hard exclusions. Tool-error markers activate refutation through ordinary input tokens, and the updated register persists in the standard cache.

## Native multi-action boundary

Ordinary text generation remains a single stream. At a tool boundary, the model exposes active slots to a shared action decoder and emits up to `K_active` tool actions in one batched decode. This is a checkpoint operation, not a set of independent agents.

The runtime executes the batch concurrently and returns for each action:

- observation tokens;
- success or failure certificate;
- optional exact canonical state key;
- cost and latency.

The next model step jointly consumes every result. It hard-merges equal certified keys, pools their support and provenance, applies verified negative updates, reallocates freed slots, and decides whether to act again or collapse to one answer.

## Merge rules

- **Hard merge:** only exact canonical keys or verifier-certified equivalence. Evidence and provenance pool; one slot becomes free.
- **Soft convergence:** learned similarity permits information exchange but retains both slots.
- **Refinement:** contradictory evidence can split a soft class because provenance is retained.
- **Hard cancellation:** only a certificate can remove support entirely.

This separation prevents a learned similarity error from deleting the only valid solution.

## Adaptive compute

Each slot has spawn, survival, action, and collapse gates. Training charges generated tokens, active slot-steps, tool calls, and wall time. A simple task should use one slot and collapse early. A difficult task may activate more slots or keep them alive across several tool rounds.

The implementation should update the register only in selected layers and fuse its low-rank operations. The present all-36-layer FP32 reference is a correctness path, not the serving design.

## Training

1. **Base-preserving transplant:** load every Qwen3-4B weight exactly; initialize the native readout at zero.
2. **Mechanism curriculum:** distill explicit frontier traces into slot coverage, verified merge, refutation, recovery, and collapse targets.
3. **Same-state counterfactuals:** one state, competing actions, real execution outcomes. Randomize action order and corrupt evidence in control rows.
4. **Parallel action SFT:** teach active slots to emit distinct useful tool calls and jointly consume their results.
5. **Selective midtraining:** train native parameters plus selected Qwen LoRA layers with strong stock-logit replay.
6. **Verifier-backed RL:** reward task success, useful diversity, recovery, and total work. Duplicate actions and invalid broad refutations receive no reward.

## Required ablations

- interference enabled versus disabled;
- exact merge enabled versus disabled;
- shuffled or false refutations;
- fixed versus adaptive active slots;
- one action versus native multi-action emission;
- same added parameters and supervision without persistent hypothesis state;
- stock Qwen language and coding retention.

The full slot-theater suite is mandatory:

| Variant | Question |
|---|---|
| 1 slot x 1024 scalars | Do multiple hypotheses matter? |
| 8 slots x 128 scalars | Proposed register |
| 16 slots x 64 scalars | Does slot count matter at fixed stored capacity? |
| Slots without interaction | Does constructive exchange matter? |
| Slots without inhibition | Does verified refutation matter? |
| Random inhibition | Is suppression semantic? |
| Shuffled failures between tasks | Is the evidence causally matched? |
| Reset every token | Does recurrence matter? |
| Reset after tool calls | Does cross-tool persistence matter? |
| Disable frontier-to-token writeback | Does the model use the register? |
| Equal-parameter ordinary FFN adapters | Is the gain explained by 36.1M extra parameters? |

The equal-parameter FFN baseline is an admission requirement, not an optional paper ablation.

## Mechanistic gates

Before a broad capability run, show that the slots behave as claimed.

### Selective inhibition

Construct tasks with identifiable hypotheses. Measure every slot before and after a verified result that disproves one hypothesis. The matching slot should fall sharply while unrelated slots remain stable or gain relative mass. Unrelated, shuffled, or false evidence must not produce the same pattern.

### Slot utilization

Report pairwise cosine similarity, effective rank, assignment entropy, occupancy, survival time, and action diversity. A score gain with redundant slots is evidence for a useful recurrent adapter, not parallel hypothesis tracking.

### Wrong-path recovery

Create source-disjoint tasks where early evidence makes an incorrect hypothesis dominant, a later tool result conclusively refutes it, and a previously weaker hypothesis becomes correct. Report recovery rate, recovery latency, and work after refutation. Compare against stock Qwen, the equal-parameter FFN, shuffled evidence, and inhibition disabled.

### Persistent working memory

Place 100, 500, and 2,000 tokens between a verified failure and the related decision. Test whether the same hypothesis remains selectively suppressed. Compare the 4 KB register against equivalent textual reminders and ordinary KV-only inference.

### Layer roles

Log read and write gate magnitudes per layer. Test all-layer updates against selected-layer updates. Look for learned specialization across perception, hypothesis construction, interaction/refutation, and final integration rather than hardcoding those roles prematurely.

## Acceptance metrics

Accuracy alone is insufficient. Report:

- task solve rate and confidence intervals;
- end-to-end latency, TTFT, and decode throughput;
- generated reasoning and action tokens;
- sequential tool rounds and total tool calls;
- active slot-steps, slot diversity, and collapse rate;
- exact merges and duplicate actions avoided;
- recovery after the leading hypothesis is refuted;
- peak memory, FLOPs, energy, and cost per solved task;
- unrelated language/coding retention;
- causal effect under every ablation above.

The primary efficiency statistic is effective cost per solved task:

`total wall time, FLOPs, energy, and tool cost / successful tasks`.

Raw tokens per second is a component measurement. The architecture wins only when reduced reasoning tokens, fewer sequential tool rounds, useful parallel actions, and earlier stopping outweigh its per-token overhead.

## Prior-art boundary

Parallel output streams, learned spawn/join, continuous superposition, adaptive latent tool reasoning, and KV-cache synthesis already exist. The candidate contribution is their missing combination: a persistent native hypothesis register with verifier-grounded signed cancellation, certified hard merging, adaptive compute, and concurrent tool actions whose results jointly update the same model state.

Do not claim novelty until a deeper mechanism-level comparison confirms that no cited system implements that complete combination. Closest current references include Multi-Stream LLMs, Adaptive Parallel Reasoning, Parallel-Synthesis, Adaptive Latent Agentic Reasoning, Reasoning by Superposition, Group Think, and The Illusion of Superposition.
