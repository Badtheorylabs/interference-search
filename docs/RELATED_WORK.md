# Related work

I searched for this on 2026-09-24, after the Countdown ablations and before the language-model and code experiments. By then I knew that the same judge did better as several arms advancing together over merged states than as one chain: 77% solved at 100 expansions against 71% for the chain at 400 on six-number problems, in 5 sequential rounds against 116. I also knew that removing the merge lost most of that gain. The next question was whether a language model needs an explicit state in place of a transcript, and I wanted to know who had already built which piece.

Each entry says what the paper did and what I took from it.

## Parallel search on Countdown

**Learning Adaptive Parallel Reasoning (APR), arXiv 2504.15466.** A 228M Llama-style model trained from scratch, with 600M and Qwen2.5-1.5B variants, learns `spawn()` and `join()` to run child threads on Countdown. It is supervised on 500k synthetic traces, then trained with GRPO. It reaches 83.4% against 60.0% for serial search (SoS+) at a 4k context, 80.1% against 66.6% at 20k total tokens, and 75.2% against 57.3% at about 5 s latency. Code, model and data are at github.com/Parallel-Reasoning/APR.

APR's threads don't share state or merge, so its benchmark and released code give the cleanest head-to-head: a frontier over merged states against APR at matched tokens. I haven't run that comparison yet.

**Stream of Search (SoS), arXiv 2404.03683.** Language models trained on serialized Countdown search traces, mistakes and backtracking included, gain 25% over models trained only on optimal paths, and they learn an internal model of state transitions. It is a ready source of Countdown search data and the SoS+ baseline APR compares against.

## An explicit state in place of the transcript

- **Atom of Thoughts, arXiv 2502.12018 (NeurIPS 2025).** Reasoning as a Markov chain. Each step breaks the current question into a DAG and contracts it into a new self-contained question, so no history is carried forward. The authors' motivation is that accumulated history wastes compute and interferes with reasoning, which matches what I saw when textual notes primed the model to retry refuted attempts.
- **The Markovian Thinker (Delethink), arXiv 2510.06557.** Reasoning in fixed-size chunks with a short carryover state, so compute grows linearly and memory stays flat. It matches long chain-of-thought RL at a quarter of the training cost at 96K thinking tokens, and Qwen3-30B-A3B and GPT-OSS-120B already do it zero-shot.
- **PENCIL, arXiv 2503.14337 (ICML 2025).** Learns to erase finished intermediate thoughts. A 25M model solves Einstein's puzzle, and on QBF it needs 649 tokens of memory where chain-of-thought needed about 151,000.
- **State of Thought, arXiv 2609.16055.** A four-number internal state and a 582-parameter controller decide which past steps to keep and when to stop, for 62.6% fewer tokens and 44.6% lower latency across 16 benchmarks. It is sequential, with no branches.
- **A State-Transition Framework for Efficient LLM Reasoning, arXiv 2602.01198 (ICLR 2026).** A linear-attention summary serves as the running reasoning state.

Together these are good evidence that a compact state saves tokens and can help accuracy. None of them runs parallel arms that share and merge states, which is the part I'm testing.

## Merging states outside exact puzzles

- **FETCH, "Don't Get Lost in the Trees", arXiv 2502.11183 (ACL 2025).** Identifies over-exploration of semantically equivalent states as a core failure of LLM tree search, and merges them by agglomerative clustering of fine-tuned sentence embeddings. It also reduces verifier variance with TD(λ) training and ensembles. This is the closest existing recipe for fuzzy merging in text, which is the biggest open problem once you leave Countdown.
- **Graph of Thoughts, arXiv 2308.09687.** Aggregation nodes merge branches by synthesizing them.
- **Transposition tables** from game search. The classic exact merge: cache results by state.

## Branching in execution

- **ParallelEnv (Tan, Zhang, Zaharia, UC Berkeley; CAIS 2026 demo).** Treats execution as a graph over immutable environment snapshots in Docker or VMs, with priority, beam or round-robin scheduling of branches. It reaches 48% on SWE-bench Pro at $0.50 per task, 15 points above single-path agents at the same budget. It doesn't mention merging identical snapshots. It is the natural substrate for the code domain, with hash-merging of identical environment states and a learned judge added on top.
- **SWE-Search, arXiv 2410.20285.** MCTS inside a software agent loop.
- **Tree-GRPO, ICLR 2026 (github.com/AMAP-ML/Tree-GRPO).** Tree-structured rollouts for agent RL beat chain rollouts with a quarter of the rollout budget on Qwen2.5-3B. It is the training recipe I'd start from for a model that searches this way natively.

## Learned pruning that holds on bigger problems

- **Learning to Search and Searching to Learn for Generalization in Planning, arXiv 2605.25720.** Relational GNN Q-functions trained on Blocksworld with fewer than 30 blocks solve 488-block instances without search. This supports the size generalization of the Countdown judge, and it suggests relational or set architectures are the right inductive bias. The judge is already a set transformer.
- **Beyond A\* (Searchformer), arXiv 2402.14083.** A transformer trained on A\* traces and then fine-tuned to shorten them solves unseen Sokoban optimally 93.7% of the time, with up to 26.8% fewer search steps than A\*. Its search-dynamics bootstrapping is a way to make a trained frontier model search less than its teacher.
- **TS-LLM, arXiv 2309.17179**, and the value-guided search line (ReST-MCTS\*, process reward models). The family my frontier-plus-judge result overlaps with most.

## Native parallel reasoning in language models

**ThreadWeaver, arXiv 2512.07843 (Meta).** Qwen3-8B with adaptive threads reaches 79.9% on AIME24 against 78.3% for a sequential baseline, with up to 1.53 times lower token latency and trie-based training that works with standard inference engines. Parallel-R1, Multiverse and Native Parallel Reasoner came before it with smaller gains. It is the current reference point for native parallelism in a real model. Its gains are modest, and it has no shared state and no merging.

## Where that leaves the claim

Every ingredient exists somewhere: explicit Markov state (AoT, Delethink, PENCIL), parallel threads (APR, ThreadWeaver), merging equivalent states (FETCH, transposition tables), execution snapshots (ParallelEnv), and pruning that generalizes (relational Q-functions). I haven't found them combined: synchronized arms over an explicit shared state space, with merging and a learned judge, for both thinking and execution. The ablation is why the combination matters. Arms without merging lose most of the gain, and a global best-first memory without synchronized depth does worse than a single chain.

## What this suggested next

1. Run APR's own Countdown setup with its released code and data: serial SoS+, APR and the state frontier at matched tokens and latency.
2. Test the state hypothesis on a real model without training. Delethink reports that Qwen3-class models reason in a Markovian way zero-shot, so transcript against state against state frontier on Qwen3-1.7B costs no training for a first read. I ran this, and it is in the research log: prompting alone did not make the small model search over states, and the structure only paid off once the environment executed moves and a judge ranked them.
3. Train the native version with existing recipes: SoS or APR-style supervised traces, then Tree-GRPO-style tree rollouts for RL.
4. For code, build on ParallelEnv-style snapshots with hash-merging and a judge, and use FETCH-style embedding merges where exact hashes don't apply.
