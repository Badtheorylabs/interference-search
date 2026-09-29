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
| Can a model maintain multiple latent states? | Latent frontier benchmark | No, not in this implementation. All K slots collapse to one state (slot cosine 1.0). |
| Can states interact? | Learned interference task | Not tested. With identical slots there is nothing to interact. |
| Can it recognize equivalent states? | Countdown multiset collapse | Not tested. The pair set has no hard negatives; a shortcut scores AUC 0.993. |
| Can it merge them? | State-collapse benchmark | Not tested. The merge update reflects slots through their mean instead of pulling them together. |
| Can it suppress invalid states? | Dead-end benchmark | Not tested. The survival gate drives the frontier to 1e-11 during training. |
| Can it advance a frontier without external search? | End-to-end reasoning benchmark | queued |
| Does it beat an ordinary transformer at matched compute? | Matched-compute baseline | Not yet tested. The last comparison gave vanilla about 3.6x the FLOPs (see audit). |
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

This run does not test the primitive. The audit below shows why, and it
also voids the earlier positive result: neither number says anything about
native latent interference.

### Audit, 2026-09-29

Tools: `audit_data.py`, `diagnose.py`, `collapse_probe.py`, and
`measure_compute.py`, run on the Spark GB10.

**The task does not require state tracking.**

- Each seed's 192 groups come from only 2 starting number sets.
- Looking up the final move token in a table of training rows scores
  0.31 to 0.40 held-out state accuracy across seeds 0 to 4. Vanilla scored
  0.31 and 0.37 on seeds 0 and 1. Vanilla performs like that lookup table.
- The model never sees the starting numbers. In 106 of 178 depth-2 test rows
  the answer contains a number that appears nowhere in the input. It can only
  be recovered by recognizing which of the two start sets the operands came
  from.
- Every depth-3 test row (206 of 206) has a one-number answer: the result of
  the final move. That is arithmetic on the last token, not trajectory
  tracking.
- 232 of 384 "held-out" trajectories have their commutative twin (a*b for
  b*a) in training.

**The equivalence test does not measure equivalence.** All 269 positive
pairs share a start set, and all 131 negatives come from different start
sets. `make_pairs` skips same-start negatives, so there are 0 hard
negatives. Counting shared operand digits, with no state tracking at all,
scores AUC 0.993.

**The model does not branch, merge, or feed back.**

- Branching: the slots start at norm 0.23, while trunk tokens have norm
  13.3. The proposal update is dominated by the token, which is the same
  for every slot, so slot updates have cosine 0.9996 at the first step.
  With merge off, the slot cosine is 0.9992 after 8 steps and stays at 1.0
  through training. The K=8 frontier is one state copied 8 times.
- Merge: the merge coefficients sum to 1.99 per slot (equiv 0.50 x alpha
  0.49 x 8 partners). The update `f_i + sum_j c(f_j - f_i)` with a sum of 2
  gives `2*mean - f_i`. That reflects each slot through the mean instead of
  pulling slots together: deviation cosine before vs after is -0.90, and
  the spread ratio is 1.01.
- Feedback: `unit.up`, `head`, and `pair_head` (17,540 params) receive no
  gradient. The unit's `x + up(mean f)` output is thrown away, so the
  frontier never writes back into the token stream.

**The full model diverges numerically.** Equal-step training, seed 0:
interference hit a max gradient norm of 9.3e10. The survival gate fell to
1e-11 and the frontier norm to 5.6e-9, so the state head reads zeros (test
accuracy 0.005). Survival-off reached a gradient norm of 8.2e4. There is no
gradient clipping and no normalization in a 15 to 23 step multiplicative
recurrence. Seed 0 in the five-seed run shows the same collapse (0.018).
The instability exists before training: at initialization, the seed-1
frontier grows from norm 0.5 to 1.2e5 over 23 tokens, while seed 0 grows
roughly linearly to 4.8. This growth also amplifies float rounding. GPU and
CPU outputs agree to 2.7e-6 at seed 0 but differ by 5.9 at seed 1, a
relative error of 2e-4 at that scale. That comes from the recurrence, not
from the hardware.

**Compute was not matched.** The runner gave vanilla 3.5x the steps on the
assumption that the unit costs 3.5x the compute. Measured fwd+bwd FLOPs:
interference/vanilla = 0.969. The 3.47x is wall time from the per-token
Python loop. Vanilla therefore got about 3.6x the FLOPs. At equal steps
(seed 0): vanilla 0.292, merge-off 0.237, survival-off 0.216, interference
0.005. Parameters are also mismatched: 858,019 vs 758,502. `run_lab.py`
now uses equal steps.

Verdict: the environment cannot distinguish state tracking from lookup, and
the model does not implement branching or merging. Fix both before running
anything else:

1. Environment: put the start numbers in the input, draw many start sets
   per seed, split held-out data by computation rather than by commutative
   twin, and include same-start hard negatives at matched depth.
2. Model: give each slot its own identity (a larger slot init or per-slot
   input projection), normalize the merge coefficients (softmax over
   partners, sum at most 1), write the frontier back into the stream,
   normalize the recurrence, and clip gradients.
3. Check the frontier trace (slot cosine well below 1, finite norms)
   before running any comparison.

## Experiment 2: search inside the forward pass

The first lab asked the model to follow one path, so there was nothing to
search. This experiment gives it the start numbers and a target, and the
network runs the search itself: `frontier_env.py`, `frontier_model.py`,
`run_frontier.py`, tests in `tests/test_frontier_lab.py`.

**Task.** Four numbers from 1 to 13 and a target from 1 to 120. Can the
target be reached using every number once? Move rules come from the paper's
solver (`interference_search.countdown.children`), with values capped at
120. All 1,820 start sets are enumerated. Train and test use disjoint start
sets (80/20), and labels are balanced per target. Measured shortcuts on the
test set: target frequency scores AUC 0.500, and nearest training start set
scores 0.651.

**Network (FrontierNet).** A frontier of K=4 latent states advances one
move at a time:

| Paper, outside the model | Here, inside the forward pass |
|---|---|
| Environment executes moves | A learned arithmetic layer maps each number pair and op to a new number vector, plus a learned legality score |
| Judge cancels dead ends | A learned survival score picks the K states that advance (top-K, Gumbel exploration in training) |
| States interact | Candidates attend to each other and to the target before being scored |
| Identical states merge | Candidates whose latent state vectors have cosine > 0.98 collapse into one, pooling their support |
| Survivors advance together | The loop runs for 3 moves; the answer is whether the kept frontier hits the target |

Numbers are never snapped back to symbols. Two paths reach the same latent
state only if the network learned to put them there.

**Training signals.** Physics (true value and legality of every move the
network expands) and outcome (for reachable examples, the likelihood that
the frontier's own path hits the target). The final hit is decoded with
detached heads, so the outcome loss cannot fake a hit. No label says which
state should survive or which states are the same.

**Reference points** (seed 0, exact moves, K=4): random pruning solves 6%
of reachable test problems, and an oracle judge solves 100%. So K=4 is
tight, and survival has to do real work.

**Ladder and compute.** Measured fwd+bwd MFLOPs per example: vanilla 95,
frontier 345, +survival 459, +interference 573, full 573. `vanilla_matched`
(d=512, 6 layers, 19M params) costs 567 MFLOPs, matching full. `vanilla_aux`
adds the same move-physics labels the frontier gets. Every model trains for
3,000 steps at batch size 128.

**Metrics, all checked against exact state identity.** reach AUC;
frontier solve rate (did the kept states really reach the target);
survival AUC (does the score rank states that can still reach the target
above dead ones); path invariance AUC (do two paths to the same state have
closer latent states than two different states); merge precision and
recall; distinct and dead states kept; physics decode accuracy.

**Bug found and fixed before the real run.** An illegal move left a
partial state (one number missing). The metrics counted it as a real
state, which could have produced false hits. True values of such children
are now zeroed, so they can never count, and `test_illegal_move_kills_true_state`
pins this. The pilot numbers from before the fix are discarded; the
five-seed run is on the fixed code.

Results: pending the five-seed run.

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
