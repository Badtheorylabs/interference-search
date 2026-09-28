# Native Interference Architecture V1

Status: frozen engineering target, 27 September 2026

## Current evidence

The first self-contained Qwen3-4B transplant has 4.059B parameters, preserves stock logits exactly before training, carries eight 128-dimensional states in its cache, and trains through ordinary tool-result tokens. On 24 held-out same-state coding choices, interference produced a +2.683 target log-probability advantage with 0.00199 KL on the zero path.

That result is a mechanism admission only. Standard generated repairs solved 11/24 with interference, 10/24 with interference disabled, and 11/24 with stock Qwen. There is no overall coding capability win.

The slot-theater probe rejected the hypothesis-tracking claim for this checkpoint:

- effective slot rank: 1.000;
- mean pairwise slot cosine: approximately 1.000;
- matched versus shuffled evidence advantage: +0.053 log-probability;
- correctly routed versus misassigned evidence advantage: +1.356;
- marker versus no-marker advantage: +3.309.

The checkpoint learned a persistent failure-routing adapter. It did not learn distinct hypotheses. Do not describe it as parallel reasoning. The next revision must pass slot diversity, semantic selectivity, and wrong-path recovery gates before multi-action scaling.

V2 added learned slot identity, occupancy, and independent token attention. It also failed: raw rank stayed 1.000 and cosine stayed 1.000.

V3 added fixed slot codes, competitive token assignment, task-conditioned probes, and lower diversity pressure. It produced a strong held-out choice margin (+2.678 with 0.00096 zero-path KL) but still failed the hypothesis gate:

- raw effective rank: 1.031;
- task-conditioned effective rank: 1.013;
- raw cosine: 0.999990;
- task-conditioned cosine: 0.999998;
- matched versus shuffled evidence advantage: +0.065.

Therefore competitive routing without slot-level hypothesis targets is insufficient. V4 must supervise coverage of distinct candidate states, use bipartite candidate-to-slot assignment, attach alive/value labels to the assigned slots, and train family-level refutation. Do not respond by increasing the geometric diversity loss.

## V4 preregistration

Candidate hypotheses are private teacher targets. At inference and evaluation, the model receives only the problem, context, and available observations. It never receives a list of candidate hypotheses to copy into slots.

The target is a permutation-invariant set. Frozen teacher representations `z_j` are matched to predicted slot representations `s_i` after the forward pass:

`pi* = argmin_pi sum_j D(s_pi(j), z_j)`

Training combines candidate coverage, assigned-slot alive/value prediction, scoped refutation, certified merge, and causal behavior. Generic diversity is a weak regularizer and cannot substitute for coverage.

### V4a current result

The first teacher-only V4a run used 32 training tasks and 8 held-out tasks with four verified code candidates each. At 300 steps it reached rank 2.33 and 59.4% candidate decoding. Continuing the same checkpoint to 600 cumulative steps produced:

- task-conditioned effective rank: 2.83, passing the greater-than-2.5 representation threshold;
- matched candidate cosine: 0.341;
- candidate decoding accuracy: 59.4%, below the greater-than-80% admission threshold;
- assigned-slot alive accuracy: 71.9%.

V4a is therefore a partial representation result, not admitted. More optimization increased rank and cosine without improving candidate decoding. The next improvement should strengthen candidate-family supervision and enlarge the task-disjoint set rather than continue the same small run.

### V4 stages

- **V4a representation:** multiple endogenous candidate hypotheses can be decoded from separate slots.
- **V4b causality:** ablating or transplanting one matched slot selectively changes its represented hypothesis.
- **V4c interference:** evidence suppresses every slot dependent on a false premise; certified equivalent survivors merge and reduce live multiplicity.
- **V4d autonomy:** candidate supervision is absent at evaluation and the behavior persists on source-disjoint tasks.
- **V5 execution:** only after V4 passes, active slots emit distinct tool calls concurrently.
- **V6 efficiency:** adaptive occupancy, sparse layer updates, BF16 state, and fused kernels.

### V4 admission targets

These thresholds are fixed before training:

| Test | Admission target |
|---|---:|
| Task-conditioned effective rank on examples with at least four hypotheses | greater than 2.5 |
| Candidate to matched-slot decoding | greater than 80% |
| Dependent versus unrelated refutation effect | greater than 3x |
| Matched versus shuffled evidence | statistically clear positive gap |
| Single-slot intervention | selectively damages the assigned candidate |
| Static identity removed | specialization remains |
| Certified equivalent-candidate merge | live multiplicity decreases |
| Merge correctness | no material loss |
| Equal-parameter ordinary FFN | V4 wins hypothesis-specific tests |

### Additional causal intervention

Transplant a matched slot between tasks at the same architectural point. A transplanted hypothesis should selectively move the recipient task's downstream belief or action toward that hypothesis. Global nonspecific degradation fails the test.

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

Report raw and task-conditioned rank/cosine separately. Task-conditioned metrics subtract each slot's mean identity contribution across tasks before measuring rank. Fixed slot codes and diversity regularizers must not count as hypothesis separation.

Intervene on individual specialized slots. Removing one slot should selectively remove or weaken the hypothesis it represents rather than degrading every candidate uniformly.

### Wrong-path recovery

Create source-disjoint tasks where early evidence makes an incorrect hypothesis dominant, a later tool result conclusively refutes it, and a previously weaker hypothesis becomes correct. Report recovery rate, recovery latency, and work after refutation. Compare against stock Qwen, the equal-parameter FFN, shuffled evidence, and inhibition disabled.

Include family-level refutation: several hypotheses share one premise, a verifier disproves that premise, every dependent slot falls, and unrelated slots remain stable. Then provide equivalence evidence for two survivors and verify that multiplicity and compute fall after merging.

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

## V4a-families: preregistered admission gates

Recorded before any family-supervised run. These gates are frozen; they will not be adjusted after results are seen. The V4a admission baseline was rank 2.83 and candidate decoding 59.4 percent on eight held-out tasks.

| Test | V4a-families admission target |
| --- | --- |
| Task-conditioned effective rank | greater than 2.5 |
| Candidate to matched-slot decoding | greater than 80 percent |
| Dependent versus unrelated refutation effect | greater than 3x |
| Matched versus shuffled semantic evidence | clear statistically significant gap |
| Single-slot intervention | selectively damages the assigned candidate |
| Static identity removed | specialization remains |
| Equivalent-candidate merge | live multiplicity decreases |
| Merge correctness | no material correctness loss |
| Capacity-matched parameter-count control | native frontier beats a plain FFN adapter |

V4a-families adds three supervised signals the prior admission run lacked:

- stratified candidate selection, so every row carries at least one passing candidate and two distinct behavioral masks;
- within-family versus between-family embedding margins, so hypotheses sharing a premise converge and hypotheses with distinct premises separate;
- equivalence collapse, so slots matched to verifier-equivalent candidates converge and agree, reducing live multiplicity.

Evaluation now also reports mean family margin and retrieval multiplicity reduction alongside rank, decoding accuracy, alive accuracy, and value error.

## Derived dataset evidence

Applying family derivation to the stored 118-task corpus produced 40 rows with at least four candidates, 64 rows with verifier-certified equivalence classes, and 57 rows with premise groups of size at least two. Outcome vectors, error classes, equivalence classes, and premise groups are computed from stored per-test verifier results, not from model self-report.

The generator expands the pool across the remaining 309 MBPP tasks. Each row carries a behavioral outcome vector, an error class for failures, an equivalence class, and a premise group derived from the verifier mask. Rows qualify only when they contain at least four candidates spanning at least two behavioral masks.

If the expanded pool cannot produce at least 40 qualifying tasks with family groups and 40 with equivalence classes, family supervision is withdrawn and V4a-families reduces to decoding-only supervision. That fallback is recorded here in advance.
