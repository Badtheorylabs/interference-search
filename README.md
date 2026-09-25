# Interference Search

Language models reason in a line. They write one token after another into a single transcript, and when a step is wrong they rewind in text and try again. I wanted to know what happens if a model searches the way a quantum computer is often described: many paths at once, with the wrong ones cancelling out.

Interference Search is my answer. The model works on explicit states instead of a transcript. Every live branch expands at once, branches that reach the same state merge into one, a small trained judge cancels dead ends, and the survivors advance together, one level at a time.

This repository has the code, the trained judge, the raw results and the full research log, including the parts that failed.

The paper is in [`paper/PAPER.pdf`](paper/PAPER.pdf), and [`paper/CLAIMS.md`](paper/CLAIMS.md) maps every number in it to the file and command that produced it.

[Watch the 30-second video: an LLM thinking in text against Interference Search on the same problem](media/interference_search.mp4)

## Results

All numbers below come from runs in this repository, on one Apple M2 laptop.

**Countdown**, 30 hard four-number problems (no two-number shortcut to the target, at most two solutions):

| System | Solved | Sequential steps |
|---|---|---|
| Qwen3-1.7B thinking in text, 1,500 tokens | 3 / 30 | about 1,400 tokens |
| The trained judge, used in one line of thought, 200 judged positions | 21 / 30 | 7.9 on the ones it solves |
| Interference Search, same judge, same 200 judged positions | 30 / 30 | 3 |

Given 1,500 judged positions, the single line also solves all 30, but needs 23.7 sequential steps on average against 3 for Interference Search.

The search solves a problem in about 5 ms on the CPU. The model thinking in text takes about 21 seconds per problem. That comparison mixes two different systems: the fast search uses the environment to list moves and a 100k-parameter judge to rank them, and does not run the language model at all.

With a judge read from Qwen3-1.7B's own hidden states instead of the trained one, the search solves 15 of 30.

On one problem, [24, 98, 19, 3] → 361, the model found `((24 × 19) − 98) + 3` at token 1,313, checked it nine more times, drifted back into wrong ideas and ran out of budget without answering. Interference Search solved it in 3 steps. That run is the video above.

**Why it compounds.** Most of what a linear reasoner writes is duplicate work. At six numbers, 831,176 possible paths collapse into 13,229 distinct states, a 63× reduction; the ratio grew 3.2 and then 5.5 times as the fifth and sixth numbers were added. The judge was trained on four- and five-number problems. On seven-number problems it cut the search 12.6× without losing a solution (10 problems). On hard six-number problems with at most 24 solution paths, the safe setting is a threshold of 0.2: 98% solved with 5.6× less search. More aggressive thresholds start losing solutions.

**Code**, 30 MBPP problems that Qwen3-1.7B gets wrong on its first greedy attempt, 1,500 generated tokens each:

| Strategy | Solved |
|---|---|
| Agent loop with the full chat history | 7 / 30 |
| Revise the latest program from its test results | 7 / 30 |
| Fresh attempts (best of N) | 8 / 30 |
| Interference Search | 9 / 30 |

Programs that behave identically on the tests merge; 83% of everything the model wrote was merged away as a duplicate. Nine against eight on 30 problems is within noise. Parallel search beat both linear strategies, and with a model this small the ideas it can propose set the ceiling.

## What did not work

I kept these because they shaped the design.

- Telling the model in its prompt which attempts had already failed made it retry them more often. In the 80 tokens after a note, a third of its attempts repeated an expression the note listed.
- Rewinding the model and resampling after a repeated attempt did not help. It regenerated the same line 16 times in a row.
- Letting parallel streams attend to each other, with no merging and no judge, learned nothing that independent streams did not.
- A prompted judge ("can you still reach the target?") answered yes to almost everything.
- A linear probe on the model's hidden states judged positions worse than a probe on simple number features. The search gains come from the structure, not from the model secretly knowing the answer.

The research log in `docs/RESEARCH_LOG.md` has every run, the external reviews I asked for, and the corrections I made after them.

## Layout

```
interference_search/        the library
  core.py                   the search loop, and a one-line-of-thought baseline over the same parts
  countdown.py              the puzzle: generation, move rules, an exact solver, attempt parsing
  judge.py                  the trained Countdown judge
  countdown_domain.py       Countdown as a domain: the environment lists moves, the judge ranks them
  program_domain.py         code as a domain: sandboxed test execution, programs merge by behaviour
weights/countdown_judge.pt  the judge used in every result above
experiments/
  countdown/                benchmark, compression, judge training, linear vs frontier and ablations
  code/                     the four code strategies on MBPP
  llm/                      the language-model runs: Phase 0 and 1, state prompting, probes, traces
  toy/                      parallel-stream policies trained from scratch (a negative result)
  visualize/                the data behind the video
tests/                      pytest
results/                    the raw outputs behind the numbers in this README
paper/                      the paper, its sources and the claims-to-evidence table
data/ media/ docs/          sanitized MBPP, the video and animation, the research log and related work
```

## Running it

Python 3.10 or newer. The search, the judge and the tests run anywhere PyTorch does. The language-model
experiments use [MLX](https://github.com/ml-explore/mlx) on Apple silicon.

```bash
pip install -e ".[test]"          # add ,llm for the language-model experiments
pytest                            # 18 tests, about 20 seconds

cd experiments/countdown
python compression.py             # how many paths collapse into states, exact
python benchmark.py               # one line of thought vs frontier vs Interference Search
python frontier_vs_linear.py --ablation
python train_judge.py             # retrain the judge and test it on bigger problems

cd ../code && python benchmark.py # the code strategies with Qwen3-1.7B (Apple silicon, about an hour)
```

Each experiment prints its table and writes a JSON file next to where it runs.

## Related work

Every ingredient exists somewhere on its own: parallel threads inside one model (APR, ThreadWeaver), branching agents over environment snapshots (ParallelEnv), merging equivalent states in tree search (FETCH, transposition tables), fixed-size reasoning state (Atom of Thoughts, the Markovian Thinker, PENCIL), and learned pruning that generalizes to bigger problems. Interference Search combines explicit states, merging, a learned judge and a level-synchronous frontier, for both thinking and execution. `docs/RELATED_WORK.md` lists the papers and what each contributed here.

## What comes next

Everything here runs without training the language model. The next step is training the architecture into a model with reinforcement learning, then measuring it on real agent tasks such as SWE-bench and Terminal-Bench at matched compute.

## Licence

Apache 2.0, see `LICENSE`. Sanitized MBPP is from Google Research under CC BY 4.0, see `data/README.md`.

Built at BTL (Bad Theory Labs) by Al-ameen, Lagos, September 2026.
