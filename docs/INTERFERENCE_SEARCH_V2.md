# Interference Search v2: scale the released mechanism

Status: local engineering prototype, 26 September 2026. The frontier cell was subsequently trained in a bounded 4B pilot and failed its inhibition gate. See the [compact-state pilot report](COMPACT_STATE_PILOT_2026-09-26.md) before using this as a training plan.

## Keep the part that worked

The release showed that a level-synchronous frontier over merged states can find solutions with fewer sequential rounds than a single chain on Countdown. Merging mattered in its ablation. I want the model to perform the proposal and judging work inside that structure, with one shared state of active branches and verified failures. The environment may execute moves and verify outcomes; it must not hand the model a list of all promising next states.

The first scaled version keeps an explicit frontier. A compact factorized register may eventually replace parts of it, but that is a further hypothesis. Training a 4B model on a new factor algebra before the current frontier becomes model-native would stack two unproven changes.

## Fix the current implementation first

The code domain used equal results on a finite list of visible tests as a hard merge key. Two programs can match those tests and still differ on unseen inputs or under a later revision. V2 uses canonical program structure plus observed behavior for safe repeat detection. It treats matching visible-test behavior as a soft cluster and keeps different representatives when the width allows. This changes what survives the frontier, not the old paper's recorded result.

The code benchmark also scheduled a full generation round before checking its token budget. In a two-problem local smoke at a nominal 640 generated tokens, the released arm averaged 1,080 tokens per problem. The capped v2 arm stopped at 640. Neither solved the two problems. This is a correctness receipt, not an efficacy comparison: the sample is tiny and the random generations were not paired. The legacy arm and its published outputs remain available.

## Model-facing computation

At each search level, exact environment keys merge identical states. A backbone encodes the task once and each surviving state into a vector. `NativeFrontierCell` receives all live state vectors in one call, lets them attend to each other, and emits action and survival scores for every state. The model can supply content-bearing action vectors; learned action slots are the bounded fallback. The environment executes selected actions and returns a new candidate pool for the next level. The task vector and unchanged state encodings can be reused across rounds.

Verified failures enter as separate vectors. For each proposed action, the cell predicts a successor vector. It compares that predicted successor with verified failed states, then subtracts a nonnegative match from the action score. It also applies a separate match to state survival. This is the missing cross-branch operation: actions from different parents predicted to reach the same failed region can be suppressed before another action is sampled. A transition loss against executed successors gives that comparison meaning; without it, similarity between an arbitrary action slot and a failed state is a weak cue. The matching path cannot increase a score. Verifier-certified hard exclusions mask states or actions separately. Similarity by itself never becomes a hard claim that neighboring states are dead. The cell has positive, zero, and negative coupling modes with the same weights and computation, so the sign can be tested without changing the model's capacity. Verified events must be deduplicated before they enter the cell.

This is still a reference module. A lightweight state-text encoder and a Qwen3 bridge connect a tiny decoder backbone to the cell. The [CPU receipt](../results/native_lm_smoke.json) and [H100 receipt](../results/native_lm_h100_smoke.json) record forward, backward, changed backbone, action, transition, and frontier weights after one optimizer step, and equal output after save/reload. A Countdown action codec maps model action slots to operand indices and an operation; its legal transitions match the existing Countdown move rules. It supplies syntax, not promising next states. Open-ended code action generation, a real training curriculum, and 4B memory and throughput remain untested. The 26 local and GPU tests establish tensor shapes, action-level inhibition, permutation behavior, gradient flow, and separation of soft and hard exclusions. They do not establish useful search behavior.

## Train the behavior the release could not test

Use exact Countdown search traces to bootstrap state and action labels, while keeping unseen problem structures out of training. The model should learn to propose diverse legal moves through the action codec, rank live states, and react to a verified dead state without replaying the transcript. Then move to a compositional program-induction family where the model must propose operations and the verifier supplies counterexamples. Countdown is a mechanism curriculum; it cannot be the only promotion test. The program-induction stage needs its own action representation and encoder before training begins.

The first 4B experiment needs three matched arms: the same backbone with a plain single-state scratchpad, with the shared frontier cell but zero refutation coupling, and with verified inhibitory coupling. Keep data, trainable parameters, optimizer work, and total measured compute aligned. Add the released cached frontier and a deterministic cached solver as system baselines. Run blank, shuffled, and sign-flipped refutation checks after training. If inhibitory coupling does not change behavior, a solve-rate difference cannot be credited to it.

Count generated and read tokens, model forward FLOPs, verifier calls, state-merge work, peak memory, and end-to-end latency. A model that needs one 4B forward pass per state may lose to the tiny released judge even if it needs fewer rounds. The cell must process the frontier together and produce enough useful proposals per pass to repay its cost.

## Promotion bar

A 24 to 48 hour H100 run is a feasibility experiment. It should prove checkpoint load, forward, backward, optimizer update, changed weights, save/reload, and held-out behavior for this exact architecture. A breakthrough claim needs a causal ablation result, source-disjoint transfer beyond Countdown, and better accuracy per total work than a cached search system using the same backbone. A positive result against a weak single chain alone would leave the central question open.

The [one-H100 plan](ONE_H100_MIDTRAIN_PLAN.md) records the hardware and Unsloth gate. The [factor-register note](NATIVE_ARCHITECTURE_RESEARCH.md) is a later path toward representing more branches than fit in an explicit frontier.
