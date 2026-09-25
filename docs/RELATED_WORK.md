# Related work that can speed this up (searched 2026-09-24)

Our current finding: the same model does about 4x better per unit of compute, with about 20x fewer
sequential steps, when it works as several linear arms that advance together over a shared, merged
state space, instead of one chain over a transcript. Merging carries most of the compute win. A
learned refuter generalizes from 4-5 to 6-number Countdown. The next hypothesis is that an LLM needs
an explicit state instead of a transcript.

Below, each piece of prior work is listed with what it gives us.

## 1. Closest: parallel search on Countdown

**Learning Adaptive Parallel Reasoning (APR), arXiv 2504.15466.** A 228M Llama-style model trained
from scratch (plus 600M and Qwen2.5-1.5B variants) learns `spawn()` / `join()` to run child threads
on Countdown. Supervised on 500k synthetic traces, then GRPO. It reaches 83.4% against 60.0% for
serial search (SoS+) at a 4k context, 80.1% against 66.6% at 20k total tokens, and 75.2% against
57.3% at about 5 s latency. Code, model and data: github.com/Parallel-Reasoning/APR.
*Use:* their benchmark, baselines and released code are the comparison table we need. Their threads
don't share state or merge, so "frontier over merged states" against APR at matched tokens is a clean
head-to-head.

**Stream of Search (SoS), arXiv 2404.03683.** Language models trained on serialized Countdown search
traces, including mistakes and backtracking, gain 25% over training on optimal paths only, and learn
an internal world model of state transitions.
*Use:* a ready source of Countdown search data and the SoS+ baseline that APR also uses.

## 2. Explicit state instead of transcript (state instead of transcript)

- **Atom of Thoughts, arXiv 2502.12018 (NeurIPS 2025).** Reasoning as a Markov chain: each step
  decomposes the current question into a DAG and contracts it into a new self-contained question
  state, so no history is carried. Their stated motivation: accumulated history wastes compute and
  interferes with reasoning. That matches our Phase 1 priming result.
- **The Markovian Thinker / Delethink, arXiv 2510.06557.** Reasoning in fixed-size chunks with a
  short carryover state, so compute is linear and memory is flat. It matches long chain-of-thought RL
  at a quarter of the training cost at 96K thinking tokens. Qwen3-30B-A3B and GPT-OSS-120B can already
  do it zero-shot.
- **PENCIL, arXiv 2503.14337 (ICML 2025).** Learns to erase finished intermediate thoughts. A 25M
  model solves Einstein's puzzle; on QBF it needs 649 tokens of memory where CoT needed about 151,000.
- **State of Thought, arXiv 2609.16055 (this month).** A 4-number internal state plus a 582-parameter
  controller decides which past steps to keep and when to stop: 62.6% fewer tokens and 44.6% lower
  latency across 16 benchmarks. Sequential only, no branches.
- **A State-Transition Framework for Efficient LLM Reasoning, arXiv 2602.01198 (ICLR 2026).** A linear
  attention summary acts as the running reasoning state.

*Use:* strong evidence that dropping history in favour of a compact state saves tokens and helps
accuracy. None of these combine the state with parallel arms that share and merge states. That
combination is our angle.

## 3. Merging states outside exact puzzles

- **FETCH, "Don't Get Lost in the Trees", arXiv 2502.11183 (ACL 2025).** Names over-exploration from
  semantically equivalent states as a core failure of LLM tree search, and merges them by
  agglomerative clustering of fine-tuned sentence embeddings. It also reduces verifier variance with
  TD(λ) training and ensembles.
  *Use:* a ready recipe for fuzzy merging in text domains, which is our biggest open problem outside
  Countdown.
- **Graph of Thoughts, arXiv 2308.09687.** Aggregation nodes merge branches by synthesis.
- **Transposition tables** from game search: the classic exact-merge mechanism, cache by state.

## 4. Execution-side branching (thinking and execution)

- **ParallelEnv (Tan, Zhang, Zaharia, UC Berkeley; CAIS 2026 demo).** Execution as a state graph over
  immutable environment snapshots (Docker or VMs), with priority, beam or round-robin branch
  scheduling. 48% on SWE-bench Pro at $0.50 per task, 15 points above single-path agents at equal
  budget. Merging identical snapshots isn't mentioned.
  *Use:* the execution substrate for our code domain. The new part we'd add is hash-merging identical
  environment states plus learned refutation.
- **SWE-Search, arXiv 2410.20285.** MCTS inside a software agent loop.
- **Tree-GRPO, ICLR 2026 (github.com/AMAP-ML/Tree-GRPO).** Tree-structured rollouts for agent RL beat
  chain rollouts with a quarter of the rollout budget on Qwen2.5-3B.
  *Use:* the RL training recipe for a frontier-structured model.

## 5. Learned pruning that generalizes to bigger problems

- **Learning to Search and Searching to Learn for Generalization in Planning, arXiv 2605.25720.**
  Relational GNN Q-functions trained on Blocksworld with fewer than 30 blocks solve 488-block instances
  without search.
  *Use:* support for our refuter's size generalization, and a hint that relational or set
  architectures are the right inductive bias. Our refuter already is a set transformer.
- **Beyond A* / Searchformer, arXiv 2402.14083.** A transformer trained on A* traces, then fine-tuned
  to shorten them, solves unseen Sokoban optimally 93.7% of the time with up to 26.8% fewer search
  steps than A*.
  *Use:* the "search dynamics bootstrapping" recipe for making our frontier model's search shorter
  than its teacher's.
- **AlphaZero-like tree search for LLMs (TS-LLM), arXiv 2309.17179**, and the value-guided search line
  (ReST-MCTS*, PRMs): the baseline family our frontier-plus-value result overlaps with.

## 6. Native parallel reasoning in LLMs

**ThreadWeaver, arXiv 2512.07843 (Meta).** Qwen3-8B with adaptive threads: 79.9% on AIME24 against a
78.3% sequential baseline, up to 1.53x lower token latency, trie-based training that works with
standard inference engines. Parallel-R1, Multiverse and Native Parallel Reasoner are its weaker
predecessors.
*Use:* the current state of the art for "native parallel" in a real LLM. Its gains are modest, and it
has no shared state or merging, which again marks where our piece fits.

## Where that leaves the novelty claim

Every ingredient exists somewhere on its own: explicit Markov state (AoT, Delethink, PENCIL),
parallel threads (APR, ThreadWeaver), merging equivalent states (FETCH, transposition tables),
execution snapshots (ParallelEnv), learned pruning that generalizes (relational Q-functions). What
we haven't found is the combination: **synchronized parallel arms over an explicit shared state space,
with merging and a learned refuter, built into the model, for both thinking and execution.** Our
ablation says the combination is where the gain lives: arms without merging lose most of it, and a
global memory without synchronized depth is worse than a single chain.

## Fastest path given this

1. **Benchmark against APR on its own Countdown setup** with its released code and data. One table:
   serial (SoS+), APR, and our state frontier at matched tokens and latency.
2. **Test the state hypothesis on a real LLM cheaply:** Delethink reports that Qwen3-class models do
   Markovian reasoning zero-shot, so transcript-linear vs state-linear vs state-frontier on Qwen3-1.7B
   needs no training for a first read.
3. **Train the native version with existing recipes:** SoS/APR-style supervised traces, then
   Tree-GRPO-style tree rollouts for RL. That's much cheaper than inventing a training scheme.
4. **For code, build on ParallelEnv-style snapshots** and add hash-merge plus refutation. Use
   FETCH-style embedding merging where exact hashes don't apply.
