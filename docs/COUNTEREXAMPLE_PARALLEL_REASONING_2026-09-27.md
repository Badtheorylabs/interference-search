# Interference Search: counterexamples that remove unseen paths

Status: controlled inference-time experiment, 27 September 2026. This tests one missing part of the released architecture: whether feedback from an executed path can eliminate many unexecuted paths while the search state stays compact. It does not involve LLM training or parallel token decoding.

The previous shared Countdown graph merged paths that reached the same environment state, but it rarely gained from proof propagation. I built a small program-search domain where a failed execution returns an exact counterexample. A program has 24 binary instruction choices, so there are 16,777,216 possible programs. Six registers execute the choices under independent, ring, or sparse dependencies. The verifier returns the first register whose output differs from the hidden target. For that register, the public instruction grammar can enumerate local choice patterns and prove which ones give the wrong output. One such proof removes every complete program containing that local pattern, including programs the search never proposed.

The search policies receive a public grammar without the hidden program or target outputs. They learn one target register output only when the verifier returns it. This is an interactive oracle setup. Ordinary synthesis tasks that disclose all input-output examples up front can be attacked directly with the factor solver; this benchmark is not evidence that hiding outputs is a general feature of program synthesis.

I implemented three policies. An exact reduced ordered binary decision diagram (ROBDD) stores the surviving version space and applies only verifier-certified exclusions. A cached factor solver stores the same register constraints and does depth-first candidate recovery. It is the main control. A full-candidate blacklist is a weak control. Each parallel batch is selected before any feedback from that batch is applied. Temporary exclusions make its candidates distinct but never enter the certified state.

## What the fixed-seed run says

The run used 30 seeds per topology, a maximum of 128 verifier calls per task, and either one or eight calls per round. All 90 tasks were solved by both the ROBDD and cached factor solver under each listed search policy. The blacklist solved 0/90 within 128 calls. The table averages over all 90 tasks. `Proposed` counts candidates constructed during selection, including unverified candidates used by the coverage scheduler. Local wall time includes selection, certificate enumeration, and the cheap simulator verifier.

| Policy | Batch | Selection | Verifier calls | Rounds | Proposed | Wall time, factor / ROBDD |
|---|---:|---|---:|---:|---:|---:|
| Cached factor and ROBDD | 1 | Lexicographic | 6.42 | 6.42 | 6.42 | 2.57 / 16.86 ms |
| Cached factor and ROBDD | 8 | Lexicographic | 44.07 | 5.73 | 44.07 | 14.34 / 17.57 ms |
| Cached factor and ROBDD | 8 | Random preferences | 35.80 | 4.88 | 35.80 | 10.43 / 16.17 ms |
| Cached factor and ROBDD | 8 | Output coverage | 34.16 | 4.67 | 244.38 | 64.43 / 32.30 ms |

These are small local CPU timings and are sensitive to implementation and run order. The robust comparison is work: diverse eight-wide batches cut serial depth from 6.42 to 4.88 rounds, but used 35.80 verifier calls instead of 6.42. Coverage searched about 244 candidates to save another 0.21 rounds. Neither scheduler improved the number of solved tasks, and the cached factor solver matched the ROBDD's verifier-call and round counts exactly.

The ROBDD did compactly represent the live set. For the diverse eight-wide runs, median peak reachable nodes were 25.5 for independent, 45 for ring, and 39.5 for sparse topology. It applied an average of 83, 1,203, and 544 forbidden local cubes respectively. The cached factor solver also avoided full-program enumeration, so compactness alone does not establish an advantage over the simple control. Ring's wider local scope made the ROBDD update substantially more expensive.

There is also a depth limit in this verifier protocol. Across the same 30 seeds, one register had an average of 13.32 possible outputs in the independent topology, 31 in the ring topology, and 24.99 in the sparse topology. The verifier reports only the first wrong register. A batch of eight cannot cover every possible output at even the first uncertain register in the average case. Most paths therefore return the same early feedback, and later registers are exposed only after another round. A model prior that concentrates the likely output, a verifier that reports more than one mismatch, or cheaper much wider batches would change that condition. The current architecture does none of those.

This is a mechanism check, not a breakthrough result. Parallel execution reduces the number of dependent verifier rounds only modestly here, and at a large increase in total calls. The ROBDD is a known classical representation; counterexample-guided synthesis and abstraction refinement are established, including [BLAZE](https://arxiv.org/abs/1710.07740). Recent [Narcissus](https://arxiv.org/abs/2608.25657) also makes a generic claim of new LLM-guided enumerative synthesis unsafe. The useful architectural requirement is sharper: a learned reasoner must produce diverse, high-value queries from a shared state and improve verified solve rate or latency **after** matching total model, verifier, selection, and memory work against cached symbolic search.

## Reproduction and acceptance boundary

The code is in [`register_program.py`](../interference_search/register_program.py), [`register_program_search.py`](../interference_search/register_program_search.py), and [`register_program_benchmark.py`](../experiments/register_program_benchmark.py). The [full per-seed receipt](../results/register_program_counterexample_30seed.json) includes all policies, topologies, batch widths, work counters, and timings. Run `.venv/bin/python experiments/register_program_benchmark.py --seeds 30 --call-budget 128 --output results/register_program_counterexample_30seed.json` from the release repository. The full local suite passed 41 tests.

The next model experiment needs an actual ambiguity that a deterministic parser or cached factor solver cannot resolve cheaply: uncertain semantic equivalence, expensive partial execution, or long-range constraints learned from examples. A candidate success must survive held-out structures, exact verification, source-disjoint tasks, and a matched cached-solver baseline. Until then, midtraining a 4B or 8B model would train against an unproven search target.
