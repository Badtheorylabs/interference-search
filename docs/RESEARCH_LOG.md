# Interference Search: Experiment Design

Status: Phase 0 and Phase 1 run locally (see results at the end).
Date: 2026-09-22

## The idea in one paragraph

Inference compute is spent today on longer chains or on wider sets of samples that never talk to each other. A quantum computer gets its advantage from a third thing: every candidate evolves together, and amplitudes can be negative, so wrong paths cancel each other out before measurement. I want the LLM version of that. K reasoning streams run inside one model at once, and when one stream proves a hypothesis false, that refutation actively pushes every sibling stream away from the same hypothesis. I'm calling the mechanism **destructive coupling**. The question is whether it buys more accuracy per token than longer chains, independent samples, or streams that can merely see each other.

## What already exists (searched 2026-09-22)

Parallel reasoning is now a crowded field. What I found splits into four groups.

**Streams that can see each other.** Hogwild! Inference (NeurIPS 2025) runs several workers over one shared KV cache, training-free, with up to 3.6x speedup. Group Think (2505.11107) has concurrent threads with token-level visibility that can hand off mid-sentence. Multi-Stream LLMs (2605.12460) trains cross-stream causal attention on Qwen models up to 27B. All three give streams positive visibility. None has an explicit mechanism for one stream to suppress another. The Multi-Stream paper says so directly.

**Native parallel branching trained with RL.** Parallel-R1 (2509.07980), ParaThinker (2509.04475), Native Parallel Reasoner (2512.07461), TSLM (2601.22688). Branches fork and merge, but they run isolated during generation and only meet at the merge.

**Central pruning.** Parallel-Probe (2602.03845) probes every branch for its current answer and kills branches that stray from the consensus, cutting total tokens 25.8% against self-consistency on Qwen3 0.6B to 8B. DeepPrune (2510.08483) removes redundant traces after observing that about 80% of parallel traces reach the same answer. HyPER (2602.06527) expands and reduces hypothesis paths. In all of these a controller outside the model decides, based on answer agreement or a reward score. The branches never learn why a sibling failed.

**Latent superposition.** "Reasoning by Superposition" (2505.12514) proves continuous thoughts can hold several BFS frontiers at once. "The Illusion of Superposition?" (2604.06374) then finds that Coconut and Soft Thinking on pretrained models do not actually superpose; only models trained from scratch do, and width matters more than depth. That result is why this design uses explicit streams first and leaves latent coupling for later.

Two results shape the design directly:

- **Parason (2608.24658)** hand-annotated reasoning traces and found that trial parallelism (competing guesses where most fail) makes up 55 to 74% of parallelizable steps, and more on harder problems. It also argues failed trials should feed later reasoning. That is the exact information destructive coupling would broadcast.
- **Reject, Resample, Repeat (2603.07887)** analyzes parallel sampling as Sequential Monte Carlo. SMC resampling is the closest classical cousin of this idea, so it is a required baseline. The difference I'm betting on: SMC moves weight between particles using a scalar score, while destructive coupling moves content (which hypothesis died, and the evidence) and the model learns what to do with it.

**The gap.** I found no work where parallel streams send each other learned, signed, content-bearing suppression signals, trained end to end. Positive visibility exists. Central pruning exists. Stream-to-stream cancellation does not appear to.

## Hypotheses

- **H1.** At matched total tokens, streams with destructive coupling beat streams with plain cross-stream visibility on trial-heavy tasks.
- **H2.** The gap grows with width K, since more siblings means more redundant failures to cancel.
- **H3.** The gap grows with problem difficulty, following Parason's finding that trial share rises with difficulty.
- **H4.** The gain is causal. Corrupting the refutation channel at eval time should hurt accuracy. If it doesn't, the model learned to ignore the channel.

## Phase 0: is there anything to cancel? (no training, about 2 days)

Before building anything I need to know whether independent streams actually waste compute repeating each other's failures. DeepPrune's 80% figure counts identical final answers, which is a different quantity.

- Run Qwen3-8B with K=8 independent samples on AIME 2025, HMMT 2025, and a recent LiveCodeBench split.
- For every failed trace, extract the approaches it tried and abandoned. Use an LLM judge and hand-check 50 traces.
- Measure the **shared-failure rate**: the fraction of abandoned approaches that at least one sibling had already abandoned earlier in wall-clock time. Also measure the tokens spent on them.

**Gate.** If shared failures account for under 10% of tokens, destructive coupling has little to remove, and the idea dies here for about two days of compute. If they're 25% or more, continue.

## Phase 1: training-free coupling (about 1 week)

Use Hogwild!-style shared-cache inference on Qwen3-8B and compare four arms at matched total tokens:

| Arm | What the streams get |
|---|---|
| A. Independent | K separate samples plus a majority vote |
| B. Visible | Shared KV cache; each stream sees the others' tokens live |
| C. Visible + ledger | B plus a shared **refutation ledger**: streams write `<refute hyp="..." evidence="..."/>` records and are prompted to avoid anything in the ledger |
| D. Visible + controller | B plus Parallel-Probe-style consensus pruning |

Arm C is the cheap textual form of destructive coupling. On code and math, refutations should be checked mechanically where possible (run the failing test, or plug the value into sympy) before they enter the ledger, so a stream can't poison its siblings with a false refutation.

This phase is a prompting result, and prompting tends to understate trained behavior. A null here is weak evidence. A clear win is strong evidence.

## Phase 2: train it (the main experiment, about 3 to 4 weeks)

**Base models.** Pilot on Qwen3-1.7B, then Qwen3-8B. Using the same family as Parallel-Probe keeps the numbers comparable.

**Architecture.** K streams with cross-stream causal attention, as in Multi-Stream LLMs, plus a ledger segment that all streams attend to. A stream opens a ledger entry with a special token.

**Cold start.** SFT on synthetic multi-stream traces where streams refute hypotheses and siblings visibly change course. This follows Parallel-R1's finding that parallel behavior needs a curriculum before RL. Generate traces with a strong teacher model and keep only those whose refutations pass mechanical checks.

**RL.** GRPO with these rewards:

- Group reward: the collapsed final answer is correct (tests pass, or exact match).
- Refutation reward: small and positive for a verified refutation that a sibling later respects. Negative for a refutation the checker rejects. This is what blocks ledger spam.
- Token cost: a penalty on total tokens across all streams, so width isn't free.

**Tasks.** The design needs tasks where refutations can be checked mechanically:

1. **Synthetic constraint search** (graph reachability, small SAT, planning puzzles) with a tunable branching factor. This gives a clean scaling curve and ties back to the superposition theory.
2. **Math**: AIME 2025/2026 and HMMT 2025, with sympy-checkable intermediate claims.
3. **Code**: a LiveCodeBench split dated after the base model's cutoff, where test execution verifies refutations.

**Training arms.** Train B (visible only) and C (visible + ledger + refutation reward) with identical data volume, compute, and seeds. Everything below hangs on that pair.

## Phase 3: latent signed coupling (only if Phase 2 wins)

This replaces the textual ledger with the literal quantum analogy. When a stream emits a refutation, a small adapter turns it into a vector r and subtracts its projection from sibling streams' residual streams at chosen layers: h ← h − α·(h·r̂)r̂, with α learned per layer. That is a learned negative amplitude. Test whether it matches the textual ledger with fewer tokens. The superposition-illusion result says fine-tuned models resist latent tricks, so I expect this phase to be hard. I only attempt it after the textual version has proven the effect exists.

## Baselines

At every budget, all arms are compared at matched total generated tokens, with wall-clock latency reported separately.

- Single long chain of thought
- Self-consistency (majority vote)
- SMC with a process reward model (2603.07887)
- Parallel-Probe and DeepPrune
- Hogwild! / Group Think, training-free
- Trained arm B (visible, no ledger)

Width K ∈ {1, 2, 4, 8, 16}. Total token budgets ∈ {8k, 16k, 32k, 64k}.

## Metrics

- **Primary:** accuracy against total tokens, as a curve. Report area under the curve and the gap between C and B at every point, with 95% bootstrap intervals over problems and three seeds.
- **Shared-failure rate** (from Phase 0): coupling should push it toward zero.
- **Approach diversity:** distinct approaches per problem, clustered by an LLM judge and spot-checked by hand.
- **Refutation precision:** the fraction of ledger entries that pass the mechanical check.

## Causal checks for H4

Run at eval time on the trained C model:

1. **Blank ledger:** accuracy should fall back to about arm B.
2. **Shuffled ledger:** refutations from a different problem. Accuracy should drop a little, since the model should learn to ignore irrelevant entries.
3. **Poisoned ledger:** false refutations of the correct approach. Accuracy should drop sharply. If it doesn't, the streams aren't reading the ledger, and any C over B gain came from something else, such as extra training signal.

## Kill criteria

- Phase 0 shared-failure tokens under 10%: stop.
- After Phase 2 at K=8, C beats B by under 2 points with overlapping intervals on both math and code: stop, and write it up as a negative result.
- The poisoned ledger leaves accuracy unchanged: the gain isn't coupling. Report it that way.

## Compute (rough estimate, not quoted)

- Phase 0: about 1 GPU-day on a single H100 for inference.
- Phase 1: about 3 to 5 GPU-days.
- Phase 2: the 1.7B pilot is about 1 node-day on 8×H100. The 8B run is roughly 4 to 8 node-days including SFT, RL, and all eval arms.
- Phase 3: unknown until Phase 2 shows what matters.

## Risks

- **RL ignores the coupling.** Streams learn to act independently and the ledger becomes decoration. The poisoned-ledger check catches this, and the refutation reward is there to prevent it.
- **Ledger spam.** Streams refute everything to farm reward. The mechanical checker and the negative reward for rejected refutations handle this, but they only work on checkable domains, which limits how general the claim is.
- **Premature cancellation.** A correct approach gets refuted early by a buggy check, and every stream abandons it. Track how often the right approach appears in the ledger. This is the failure mode quantum interference doesn't have, because amplitudes don't lie.
- **The gain is just more supervision.** C gets an extra reward signal that B doesn't get. To control for this, add an arm B+ that receives the same refutation reward for writing refutations to a private ledger that no sibling can read.

## Sources

- Hogwild! Inference: https://arxiv.org/abs/2504.06261
- Group Think: https://arxiv.org/abs/2505.11107
- Multi-Stream LLMs: https://arxiv.org/abs/2605.12460
- Parallel-R1: https://arxiv.org/abs/2509.07980
- ParaThinker: https://arxiv.org/abs/2509.04475
- Native Parallel Reasoner: https://arxiv.org/abs/2512.07461
- TSLM: https://arxiv.org/abs/2601.22688
- Parallel-Probe: https://arxiv.org/abs/2602.03845
- DeepPrune: https://arxiv.org/abs/2510.08483
- HyPER: https://arxiv.org/abs/2602.06527
- Reasoning by Superposition: https://arxiv.org/abs/2505.12514
- The Illusion of Superposition?: https://arxiv.org/abs/2604.06374
- Parason: https://arxiv.org/abs/2608.24658
- Reject, Resample, Repeat: https://arxiv.org/abs/2603.07887
- Parallel-Synthesis (KV-cache branch merging): https://arxiv.org/abs/2606.14672

## Phase 0 metric, fixed before the full results (2026-09-23, 4/30 problems in)

Local run: Qwen3-1.7B 4-bit on MLX, Countdown with 4 numbers, K=8, 2,048 tokens per stream.
Primary gate metric: share of all generated tokens spent on failed attempts a sibling had
already refuted earlier in token time, with each stream's first attempt costed at that stream's
median attempt cost. The strict (first attempt = 0) and loose (first attempt = full preamble)
variants are reported alongside it but do not decide the gate. Self-repeats are counted
separately and never as sibling-redundant.
Middle zone (10% to 25% on the primary metric): run the cheap training-free Phase 1 ledger test
locally, but do not start Phase 2 training unless Phase 1 shows a gain.

## Results so far (2026-09-23, local, Qwen3-1.7B 4-bit, Countdown, K=8)

**Phase 0 (6 problems, stopped early; Phase 1's arm A extends it).** 381 failed attempts;
29.7% repeated a sibling's earlier refutation, 45.1% repeated the stream's own. Primary token
metric 16.5% (middle zone), so Phase 1 ran as pre-registered.

**Phase 1 (24 problems, 256-token chunks, 8 rounds, 16,384 generated tokens per group).**

| Arm | Found | Mean tokens to find | Sibling-redundant failures | Self-repeats |
|---|---|---|---|---|
| A independent | 22/24 | 5,803 | 23.0% | 55.1% |
| P private notes | 21/24 | 6,485 | 21.7% | 55.6% |
| C shared notes | 22/24 | 5,376 | 27.3% | 54.3% |

C vs P: C found 1 problem P missed, and found earlier on 3 of 3 shared divergent problems.
Sign-test p = 1.0. Not significant. 16 of 24 problems were decided by round 1, before notes
could act, so the benchmark sat near the ceiling.

Mechanism check failed. Within 80 tokens after a note, 33.6% of C's attempts repeated a sibling
refutation, against 21 to 22% in A and P. The model re-checks what the note lists, so a textual
"do not retry" primes the refuted expressions. Private reminders did not lower self-repeats either.

Conclusion: prompt-level ledgers do not produce destructive coupling in a 1.7B model. The next
test moves suppression into decoding: a stream that writes an already-refuted attempt is rewound
to before it and resampled, with discarded tokens charged to its budget (Phase 1b).

**Phase 1b (rewind-and-resample, 5-number Countdown), stopped after 5 problems.** Uninformative
by construction: the 1.7B model wrote 0 to 5 complete 5-number attempts per problem, so there was
almost nothing to suppress, and the three arms tracked each other exactly. One observation is
worth keeping. On problem 2 the self-rewind arm rewound 16 times and the model regenerated the
same refuted expression each time: resampling from the same context reproduces the failure,
because the cause sits upstream in the context. Deleting a bad attempt at decode time does not
change what the model was going to do. This is the practical case for putting suppression in the
weights (Phase 2/3) rather than in prompts or decoding.

## Toy from-scratch test (started 2026-09-23)

Code in `toy/`. Hidden-goal tree search (branching 3, depth 4, 81 leaves), noisy hints shared by
all streams so streams repeat each other's mistakes, 24 lockstep steps, success when any stream
reaches the goal. Hand-coded bounds at noise 1.5: greedy stays at 0.40 for K = 1 to 8 (all streams
make the same mistakes); a perfect coordinator reaches 0.58 / 0.78 / 0.94 at K = 2 / 4 / 8.
Four small transformer policies (about 110k to 120k parameters), each pretrained by imitating the
independent greedy expert (no coordination in the teacher) and then trained with grouped
REINFORCE on the group reward at K = 4:

- indep: streams see only themselves
- visible: full cross-stream attention
- unsigned: visible plus a dead-end head that shifts child logits by a free-sign amount
- signed: the same head constrained to only lower logits of children resembling known dead ends

unsigned and signed start as the same function; the sign constraint is the only difference.

## External reviews (2026-09-23) and what changed

Two independent LLM reviews of REVIEW_BRIEF.md. Accepted points:

- The first toy runs were confounded: one Adam optimizer carried imitation-scale second moments into RL, and the imitation teacher (deterministic, own memory only) trained visible-capable arms to ignore siblings. Unsigned and signed runs were stopped before finishing.
- The "coordinator" is not an attainable ceiling: it lets later streams see same-step choices of earlier streams. It is now labelled a heuristic (it is essentially virtual loss from parallel MCTS). The reference for the value of sharing is soft_shared, a simultaneous softmax-greedy player.
- The sign constraint was not the only difference between signed and unsigned (softplus gives about 8x smaller effective step size on the strength at init, and the unsigned coefficient never became positive). The pathway is not globally sign-constrained, since dead events also enter through token embeddings. A proper ablation matrix is required before any sign claim.
- The project keeps the name Interference Search. The technical claim must stay precise (parallel streams share verified dead ends through a learned inhibitory pathway) until ablations show the negative coupling itself does the work, and it must be positioned against diverse beam search, virtual loss, tabu search, SemDiD, unlikelihood training, DExperts and classifier-free guidance.
- The Phase 0/1 conclusion "a pretrained model cannot be prompted or decoded into destructive coupling" overreaches. Supported version: negative textual reminders and late full-expression rewind failed for 4-bit Qwen3-1.7B on this setup.

One point partly overturned by data: with stochastic players, sharing dead ends is worth +8 points at K=4 in the leaf-only environment (soft_self 0.62, soft_shared 0.70); the zero value came from deterministic greedy. With subtree refutation p=0.25 the prize is +12 at K=4 (0.70 vs 0.82) and +14 at K=8 (0.78 vs 0.92). Learned indep in the old runs (0.54 at K=4) sat below soft_self, so training, not the idea, was the bottleneck.

Gate now running (toy/runs2): indep vs visible trained from scratch, fresh Adam, refute_p 0.25, near-uniform initial policy. If visible reaches soft_shared while indep stays near soft_self, plain attention already learns sharing in this toy and the signed head needs a harder test.

## Merged-state search with a learned refuter (2026-09-23, frontier/)

Direction change after two more reviews: the achievable form of the idea is exponential path
compression (reason over merged states) plus state-level refutation (learned bounding), not
generic exponential parallelism.

**Compression in Countdown (exact DP, 20 random solvable instances per size).** Paths per distinct
state at the last level: 3.6x (4 numbers), 11.4x (5), 62.8x (6). With 6 numbers, 831,176 paths
collapse into 13,229 states. After two operations most states are dead, and more than 80% of all
full paths pass through a dead state.

**Learned refuter, seed 0.** Set transformer over the numbers in a state plus the target, trained on
632,279 exactly labelled states from 4- and 5-number problems (2.6% alive). Tested by merged-state
BFS on 100 unseen 6-number problems, a size never seen in training:

| Method | Solved | States expanded | vs no pruning |
|---|---|---|---|
| No pruning | 100% | 21,220 | 100% |
| Hand bound (max reachable value) | 100% | 20,754 | 97.8% |
| Learned, tau = 0.2 | 100% | 5,050 | 23.8% |
| Learned, tau = 0.5 | 100% | 3,518 | 16.6% |
| Exact oracle | 100% | 223 | 1.1% |

Path-by-path search on these problems means about 294,000 expansions, so merged states plus the
learned refuter do roughly 84x less work. Caveats: one seed; the refuter costs more per state than a
C bound, so the value is inside a reasoning model where each expansion is expensive; test problems
may have several solutions. Running now: higher thresholds, a hard set with few solution paths,
7-number problems, and a saved model.

**K-stream gate result (toy/runs2, seed 0, 1,500 RL iterations from scratch, refute_p 0.25, 4,000-episode eval).**
indep 0.337 / 0.508 / 0.714 and visible 0.349 / 0.511 / 0.721 at K = 2 / 4 / 8; redundancy 43% in both.
A tie, and both sit below the simple own-memory softmax player (0.70 at K=4) and far below shared
memory (0.82). Plain cross-stream attention did not learn to share dead ends, and small-transformer RL
on this toy is weak overall. The K-stream line is paused in favour of merged-state search with a
learned refuter; the GPU package (toy/run_gpu.sh) can finish the full arm matrix cheaply if needed.

**Linear vs frontier at matched compute (frontier/native_loop.py).** Same learned viability model
for every method; cost = expansions (the model thinks about one state, the environment executes its
moves). Frontier width = budget / depth. Solve rate on unseen problems:

| 6 numbers (100) | 10 | 20 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| linear (chain, restart on failure) | 3% | 14% | 26% | 40% | 48% | 71% |
| linear + refute | 6% | 12% | 28% | 43% | 68% | 83% |
| frontier (merged) | 17% | 33% | 61% | 77% | 88% | 94% |
| frontier + refute | 17% | 33% | 61% | 77% | 87% | 94% |

7 numbers (30): at 50 expansions frontier 53% vs linear 17%; at 400 both about 87 to 90%.
Sequential rounds when solved: frontier 5 (6 numbers) and 6 (7 numbers); linear 116 and 153.

Reading: organising the same model as a merged frontier needs roughly 4x fewer expansions for the
same solve rate at 6 numbers (77% at 100 vs 71% at 400) and about 20x fewer sequential steps. The
hard refute threshold adds nothing on top of ranking by viability, which already drops those states.
Merging is not yet isolated (needs a frontier-without-merge arm), and a value-ranked frontier is close
to value-guided beam search, which is prior art; the new part to argue is the native, learned version
of merge and refute.

**What makes the frontier win (frontier/ablate.py).** 6 numbers, 100 problems, solve rate at 50 / 100 / 400
expansions: linear 26 / 40 / 71%; frontier with merge 61 / 77 / 94%; frontier without merge 39 / 50 / 74%;
best-first with a ranked memory of every seen state 1 / 10 / 57%; best-first without merge 0 / 2 / 13%.
7 numbers (30): frontier 53 / 70 / 87%, frontier without merge 37 / 40 / 70%, best-first 0% throughout.

Reading: merging carries most of the compute win (duplicates otherwise fill the width). A global
ranked memory of alternatives is worse than a linear chain: it keeps choosing shallow states that look
safe and never commits to depth. What works is several linear arms that advance one level together
over a shared state space, merge where they land on the same state, and are compared only against
states at the same depth. Next hypothesis: for LLMs the missing piece is an explicit
state instead of a transcript; test transcript-linear vs state-linear vs state-frontier on Qwen3-1.7B.

## Refuter robustness and the real-LLM state test (2026-09-24)

**Refuter on hard problems (<= 24 solution paths, 100 unseen 6-number).** tau 0.2: 98% solved at 17.9%
of states; tau 0.5: 78% at 7.3%; tau 0.7: 43% at 4.3%. The earlier "100% at tau 0.7" relied on random
problems having many solutions. The safe operating point on hard problems is tau 0.2 (5.6x less search).
**7-number problems (10, random):** tau 0.7 solves all 10 at 7.9% of states, so size generalization holds.

**Qwen3-1.7B, 30 hard 4-number problems, 1,500 generated tokens.** Transcript chain-of-thought with
thinking: 3/30. The first state and frontier arms were stopped after traces showed three flaws: with no
memory the model repeated the identical 3-step path 8 times; proposals were reflexive (it adds the first
two listed numbers and almost never multiplies); candidate lists were cut by the token cap. After adding
dead-state memory, number shuffling and a larger cap (6-problem smoke test): state with memory 0/6;
environment enumerates moves and the prompted LLM judges P(yes) 1/6; environment enumerates and the trained
refuter judges 5/6 with no LLM tokens. The prompted judge says yes to everything (mean P(yes) 1.000 for both
live and dead states, AUC 0.47).

Conclusion: a small model cannot be prompted into state-based search; its competence lives in its native
thinking mode. Next: train the judge into the LLM's weights (LoRA on exact solver labels), then compare the
environment-enumerates frontier with the trained LLM judge against transcript chain-of-thought.

**Linear probes on frozen Qwen3-1.7B hidden states (probe.py), with a surface-feature control.**
Trained on 1,376 labelled states from 60 hard 4-number problems (1:3 alive:dead), tested on unseen problems.
AUC 4-number / 5-number: prompted P(yes) 0.58 / 0.62; probe on layer 28 0.87 / 0.94 (layers 14 and 21
similar); linear probe on hand number features with the same training states 0.94 / 0.92; small refuter
0.98 / 0.97. As the judge in an environment-enumerates frontier on the same 30 hard problems:
chain-of-thought 3/30, prompted judge 4/30, LLM probe 15/30, hand-feature probe 22/30, refuter 23/30.

Correction to an interim reading: the LLM probe does not show the model "knows but can't say". Simple
number features judge better with the same data. What holds: the structure (environment executes, states
merge, a small judge ranks a level-synchronous frontier) lifts solves from 3 to 15-23 of 30 in 3 rounds,
even with a crude judge. For open domains, where no hand features exist, the judge will need training;
frozen LLM features are usable but weaker.

## Assembled system (isearch/) and the second domain (2026-09-24)

`isearch/core.py` holds one Interference Search loop (explicit states, propose, execute, merge key, judge,
level-synchronous frontier) plus a linear-chain baseline over the same parts. Domains: Countdown
(`countdown_domain.py`, environment enumerates moves, learned refuter judges) and code
(`code_domain.py`, Qwen3-1.7B writes and revises programs, the interpreter runs the tests, programs with
identical behaviour on the tests merge, tests passed is the judge).

**Countdown through the assembled core, hard 4-number set (30):** solve rate at 50 / 150 / 500 judged states:
linear chain 27 / 63 / 80%; frontier without merge 13 / 97 / 100%; Interference Search 23 / 97 / 100%.
Rounds at the largest budget: 23.7 vs 3.0. The hard 6-number run used too small a budget (1,500 judged
states solved about nothing for any arm) and was stopped; rerun with larger budgets.

**Code, 30 MBPP problems Qwen3-1.7B fails on its first greedy try, 1,500 generated tokens each:**
transcript agent loop 7/30 (stopped at round 54, no new solve after round 3); linear state refinement 7/30;
best-of-N 8/30; Interference Search 9/30, with 83% of generated programs merged as behaviour duplicates.
9 vs 8 on 30 is within noise; per-problem outcomes were not saved, so no paired test is possible. Reading:
parallel beats linear in both domains; in code the ceiling is the small model's ideas, since revision rarely
rescues a wrong idea. Next levers: a stronger proposer, or a trained judge (for example CLM-v0.1-8B heads).
Run notes: three MLX out-of-memory crashes (a concurrent trace process, growing transcript prompts, then MLX
buffer cache growth); fixed with token-sized batches, mx.set_cache_limit and mx.clear_cache.
