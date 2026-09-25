# Interference Search

Language models reason in a line. They write one token after another into a single transcript, and when a step goes wrong they rewind in text and try again. Interference Search reasons over explicit states instead. Every live branch expands at once, the environment executes the moves, branches that land on the same state merge into one, a small trained judge drops the ones that can no longer reach the goal, and the survivors advance together one level at a time.

The name comes from quantum search, where wrong paths cancel each other out. The method itself is classical and claims no quantum speedup.

This repository has the code, the trained judge, every raw result and the research log, failed experiments included. The paper is [`paper/PAPER.pdf`](paper/PAPER.pdf), also on [badtheorylabs.com](https://www.badtheorylabs.com/papers/interference-search). [`paper/CLAIMS.md`](paper/CLAIMS.md) maps each number in it to the file and command that produced it.

[The 30-second video](media/interference_search.mp4) shows Qwen3-1.7B thinking in text next to Interference Search on the same problem.

## Why Countdown

Countdown gives you a few numbers and a target and asks for an expression that uses every number once. The arithmetic is easy. Each step combines two small numbers, and in the search arms the environment does that arithmetic, not the model. The hard part is choosing which two numbers to combine, with which operation, in which order. Four numbers give about 570 complete move sequences per problem on average; six give 831,176, and after two moves more than 80% of them already run through a state that can't reach the target.

That choice is what a linear reasoner spends its tokens on, and it's the part this method changes. When Qwen3-1.7B failed, it wasn't getting sums wrong. In the first experiment, 45% of its failed attempts repeated an expression it had already ruled out itself. And on one problem it wrote the correct answer at token 1,313 and never committed to it. Countdown also has an exact solver, so every state can be labelled alive or dead without an LLM grading it. Stream of Search and APR use it to study search in language models for the same reasons.

## Results

Everything here ran on one Apple M2 laptop with 16 GB of memory. All runs are one seed.

**Countdown**, 30 hard four-number problems (no single operation lands within 25 of the target, and at most two solutions exist):

| System | Solved | Sequential steps |
|---|---|---|
| Qwen3-1.7B thinking in text, 1,500 generated tokens | 3 / 30 | one token at a time |
| The trained judge in one line of thought, 200 judged positions | 21 / 30 | 7.9 on the problems it solves |
| Interference Search, same judge, same 200 judged positions | 30 / 30 | 3 |

With 1,500 judged positions the single line also solves all 30, but it takes 23.7 sequential steps on average against 3.

The search takes about 5 ms per problem on the CPU and the model thinking in text takes about 21 seconds. Those are different systems: the fast one uses the environment to list moves and a 100k-parameter judge to rank them, and never runs the language model. When I swapped the trained judge for a linear probe on Qwen3-1.7B's own hidden states, the search solved 15 of 30.

The video is problem [24, 98, 19, 3] → 361. The model wrote `((24 × 19) − 98) + 3` at token 1,313, checked it nine more times, drifted into other ideas and ran out of budget without answering. Interference Search solved it in 3 steps.

**Most of the work a line does is duplicate.** At six numbers, 831,176 paths collapse into 13,229 distinct states, 63 paths per state. The ratio grew 3.2 times from four numbers to five and 5.5 times from five to six, so merging pays more as problems grow. The judge was trained on four- and five-number problems. On ten seven-number problems it cut the search 12.6 times and lost no solutions. It is less safe on hard problems: on six-number problems with at most 24 solution paths, a threshold of 0.2 solves 98% with 5.6 times less search, and higher thresholds start losing solutions.

**Code**: 30 MBPP problems Qwen3-1.7B gets wrong on its first greedy attempt, 1,500 generated tokens each.

| Strategy | Solved |
|---|---|
| Agent loop with the full chat history | 7 / 30 |
| Revise the latest program from its test results | 7 / 30 |
| Fresh attempts (best of N) | 8 / 30 |
| Interference Search | 9 / 30 |

Here a state is a program, and two programs that behave the same on every test merge. 83% of the programs the model wrote were duplicates of behaviour already seen. Nine against eight on 30 problems is within noise. With a model this small, revision rarely rescues a wrong idea, so the ideas it proposes set the ceiling.

## What did not work

These shaped the design, so I kept them.

- Telling the model in its prompt which attempts had already failed made it retry them more often. In the 80 tokens after a note, a third of its attempts repeated an expression the note listed.
- Rewinding the model and resampling after a repeated attempt regenerated the same line 16 times in a row. The cause sits earlier in the context, so resampling the same context brings it back.
- Parallel streams trained from scratch that could attend to each other scored 0.511 against 0.508 for streams that couldn't. Visibility alone taught them nothing.
- A prompted judge ("can you still reach the target?") answered yes to almost everything, with an AUC of 0.58.
- A linear probe on the model's hidden states judged positions less well than a probe on plain number features (AUC 0.87 against 0.94). I first read the probe as the model knowing more than it says. The feature control ruled that out, and the gains belong to the search structure.

[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) has every run in the order it happened, the outside reviews I asked for, and what I changed after them.

## Layout

```
interference_search/        the library
  core.py                   the search loop, and a one-line baseline over the same parts
  countdown.py              the puzzle: generation, move rules, an exact solver, attempt parsing
  judge.py                  the Countdown judge model
  countdown_domain.py       Countdown as a domain: the environment lists moves, the judge ranks them
  program_domain.py         code as a domain: sandboxed test runs, programs merge by behaviour
weights/countdown_judge.pt  the judge behind every Countdown result
experiments/
  countdown/                benchmark, compression, judge training, frontier against line, ablations
  code/                     the four code strategies on MBPP
  llm/                      the language-model runs: prompting, decoding, state arms, probes, the trace
  toy/                      parallel-stream policies trained from scratch (a negative result)
  visualize/                the data behind the video
tests/                      pytest
results/                    raw outputs behind the numbers above
paper/                      the paper, its LaTeX source and the claims table
data/                       sanitized MBPP
docs/                       the research log and related work
media/                      the video and its animation
```

## Running it

Python 3.10 or newer. The search, the judge and the tests run anywhere PyTorch does. The language-model experiments use [MLX](https://github.com/ml-explore/mlx) and need Apple silicon.

```bash
pip install -e ".[test]"          # use ".[test,llm]" for the language-model experiments
pytest                            # 18 tests, about 20 seconds

cd experiments/countdown
python compression.py             # paths against states, computed exactly
python benchmark.py               # one line, frontier without merging, Interference Search
python frontier_vs_linear.py --ablation
python train_judge.py             # retrain the judge and test it on bigger problems

cd ../code && python benchmark.py # the code strategies with Qwen3-1.7B, about an hour on an M2
```

Each experiment prints its table and writes a JSON file where it runs.

To check every Countdown number on any machine, including a Linux VM with no GPU:

```bash
bash scripts/reproduce_countdown.sh     # tests, compression, benchmark, ablation; about 10 minutes on CPU
```

It writes fresh outputs to `repro/` and compares them against `results/countdown`. Solve rates and step counts are deterministic, so they should match exactly; timings will differ by machine. `RETRAIN=1` also retrains the judge from scratch. With Docker:

```bash
docker build -t interference-search . && docker run --rm interference-search
```

The language-model experiments use MLX and don't run on Linux yet.

## Related work

Each piece has been done before on its own. APR and ThreadWeaver run parallel threads inside one model. ParallelEnv branches agents over environment snapshots. FETCH and transposition tables merge equivalent states in tree search. Atom of Thoughts, the Markovian Thinker and PENCIL keep a compact reasoning state in place of the full history. Relational Q-functions learn pruning that holds on bigger problems. What I haven't found elsewhere is the combination tested here: explicit states, merging, a learned judge and a frontier that advances level by level, used for both thinking and execution. [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) goes through each paper and what I took from it.

## Next

None of this trains the language model. The next step is training the search into one with reinforcement learning, then testing it on agent tasks such as SWE-bench and Terminal-Bench at matched compute.

## Licence

Apache 2.0, see [`LICENSE`](LICENSE). Sanitized MBPP is from Google Research under CC BY 4.0, see [`data/README.md`](data/README.md).

Al-ameen, BTL (Bad Theory Labs), Lagos, September 2026.
