# Native Latent Interference Roadmap

Mechanism name: **native latent interference**. Not "native reasoning"; that
claim returns only if the primitive works.

## Foundation and retraction

Foundation: the Interference Search paper. Its algorithm: keep many explicit
states alive, execute moves, merge identical states, cancel dead ends, advance
the surviving frontier together. Proven externally: Countdown 30/30 vs 21/30
for linear search at equal budget.

The research question is now exactly one sentence:

> Can the algorithmic operations of Interference Search be compiled into the
> architecture of a neural model, so they happen inside the forward pass?

Retracted and deleted from the roadmap: the V4 frontier-slot line (V4/V4a/V4b
inside Qwen3-4B). It was an experiment that failed its intended objective. It
is not a foundation and nothing builds on it. The 68.5% decoding result, the
80% gate, and the uncontrolled retrieval-knockout number are not evidence for
anything going forward. The scripts remain in `experiments/` marked archived;
the checkpoint stays frozen and untouched. The earlier "distill search into
Qwen3-1.7B" plan is also superseded: the primitive must be discovered before
anything is distilled into it.

## The four primitives

Interference Search has operations. Each becomes a learned component inside
the model, not a Python controller.

1. **Latent branching.** Model-internal frontier `F = [s₁..sₖ]`, latent
   reasoning states maintained across computation. Not K separate sequences,
   not K model calls.
2. **Interference.** `I(sᵢ, sⱼ)` learned relations: supports, contradicts,
   duplicates, depends-on, independent. More than plain attention: attention
   asks what information should flow, interference asks which hypotheses
   survive, merge, or die.
3. **State merging.** A learned equivalence operator. When two trajectories
   arrive at the same underlying state, collapse to one latent state instead
   of carrying both. This is the paper's central insight translated, and the
   hardest piece: easy for exact states (arithmetic multisets), hard in
   general.
4. **State survival.** `qᵢ = survival(sᵢ | context)`. The frontier evolves
   internally; no external judge decides which state 7 to discard.

Architecture sketch per block:

```
tokens ──→ attention ──→ MLP ──→ interference unit ──→ next
                     frontier F: branch, interfere, merge, survive
```

Language is the interface; the latent frontier is the reasoning substrate.
Attention stays; interference joins it as a second native computation.

## Environments: exact state identity only

The model must be tested where ground-truth state identity is known, so
collapse can be measured without philosophical argument.

- **Countdown multiset.** Different operation sequences reach the same
  multiset. Exact identity, already in the repo.
- **Graph search.** Different paths reach the same node. Exact identity.
- **MBPP behavior.** Different programs with identical verifier vectors.
  Existing MBPP infrastructure; comes after the first two work.

## Falsifiable questions

| Question | Experiment | Status |
|---|---|---|
| Can a model maintain multiple latent states? | Latent frontier benchmark | Unknown. First measured advantage was a memorization artifact (see retraction below). |
| Can states interact? | Learned interference task | Not demonstrated. |
| Can it recognize equivalent states? | Countdown multiset collapse | Unknown pending corrected run; threshold artifact removed. |
| Can it merge them? | State-collapse benchmark | Not demonstrated; merge-off ablation tied with interference. |
| Can it suppress invalid states? | Dead-end benchmark | Not demonstrated; survival gate needs schedule or objective change. |
| Can it advance a frontier without external search? | End-to-end reasoning benchmark | queued |
| Does it beat an ordinary transformer at matched compute? | Matched-compute baseline | Unknown pending corrected run (step budget matched to unit cost). |
| Does it scale? | 100M → 500M → 1B → 4B | forbidden until rows above pass |

Lab run 2026-09-29 (`experiments/lab/results/lab_*.json`): 858k vanilla vs
758k interference params, 400 steps, held-out trajectories within states.

**Retraction (same day).** Two bugs invalidated every number above the line
before the next section. First, the environment emitted duplicate trajectories
(pair-ordering variants of the same move sequence), so identical trajectories
leaked into train and test: the single-seed and five-seed results above
measured memorization, not state identity. Second, the equivalence metric
picked its threshold by best-of-86 fitting on the evaluation pairs, inflating
every equivalence score. Both fixed: trajectory dedup in `envs.py`, and the
threshold-free AUC metric in `evaluate_equiv` (probability a true same-state
pair outranks a true different-state pair). The deeper environment (depth 3,
192 groups, ~2000 trajectories) replaced the old shallow one. Tests:
`tests/test_interference_lab.py`, 10 passed on the H100.

Corrected five-seed run, depth-3 environment, 400 steps each: all equivalence
AUC scores tied at chance (0.276 identical across modes), state accuracy
low, survival-off collapsed. Undertrained, no signal either way. The corrected
experiment is 2000 steps for interference variants and 7000 for vanilla
(matched total compute against the unit's ~3.5x per-step cost).

Corrected five-seed run completed 2026-09-29 (`/tmp/lab_seeds4.log`, averages):

| System | exact state acc | equivalence AUC |
|---|---|---|
| vanilla | **0.340** | **0.780** |
| interference | 0.168 | 0.758 |
| merge-off | 0.188 | 0.765 |
| survival-off | 0.040 | 0.862 |

Bottom line, stated honestly: the interference unit currently **loses** to the
matched vanilla transformer on both metrics. Vanilla trains to ~zero training
loss; interference variants stall (final losses 0.13–3.8), meaning the frontier
path is an optimization burden at this budget, not a win. The survival-off
equivalence score of 0.862 comes from a diverged model (state accuracy 0.04,
loss 3.8) and is not trustworthy: a broken state head is not evidence of
equivalence ability. No primitive is demonstrated by this configuration.

Diagnosis queued: the frontier readout fails to train while direct readout
converges. The next step is a diagnostic run that logs the interference unit's
internal stats (survival means, support magnitude, merge magnitude, frontier
norm) across training to see whether the frontier collapses, saturates, or
never receives useful gradient, before changing any architecture.

## Rules

- Smallest possible system first. The lab is ~0.5M–few-M parameters, CPU/GPU
  seconds per run. Nothing at 100M+ until primitives pass at lab scale.
- Every experiment has a vanilla transformer baseline with matched parameters
  and matched compute. No matched-compute comparison, no claim.
- Outcome-only supervision. No external judge, search loop, or label tells
  the model when to merge or cancel; it must learn that from data.
- Ablations are required: merge-off and survival-off, so any gain is
  attributed to the named operator rather than extra parameters.
- Scale gates: 100M → 500M → 1B → 4B only after end-to-end frontier
  advancement beats matched-compute baselines on at least two exact-state
  environments.
- Do not attach the primitive to pretrained Qwen as the first proof. A small
  model trained from scratch where the primitive works beats a big model
  where it is hoped for.

## Lab layout

`experiments/lab/` holds the whole laboratory: environment generators with
exact state identity, the two model classes (vanilla and interference), the
matched-compute runner, and collapse metrics. Results go in
`experiments/lab/results/` as JSON, and this document's status column updates
only from runs that actually happened.
