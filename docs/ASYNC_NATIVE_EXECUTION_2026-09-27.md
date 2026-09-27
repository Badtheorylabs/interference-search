# Interference Search: parallel search and execution in the model runtime

Status: inference architecture and speed preflight, 27 September 2026. The immediate objective is to shorten end-to-end work, including model calls and executions. Better answers can follow from training the native heads later. The measurements here establish specific runtime mechanisms; they do not establish a new reasoning capability.

The old shared graph waited for every result at a search depth before making the next model call. A slow execution held up a fast path. The new [`AsyncExecutionSearch`](../interference_search/async_execution.py) starts a model batch from ready states while other executions are still running. It runs bounded tool calls concurrently, prioritizes ready states and queued executions by the native action head's score, merges equal canonical states, coalesces identical in-flight executions when the domain provides an exact execution key, and cancels pending work once a verified goal arrives. The model and executor have separate capacity limits. The scheduler counts proposals, execution requests, actual calls, merges, cancellation, and wall time.

The [`NativeAsyncCountdownDomain`](../interference_search/native_async_countdown.py) connects that scheduler to `NativeFrontierLM`. A Qwen3 backbone processes the task once; the native frontier head proposes legal action slots for each ready batch and predicts successor vectors. Exact execution decides the resulting state and the goal. Each predicted vector is paired with its observed successor for later training. Certified terminal failures enter the refutation context. The current head is randomly initialized, so this is a functioning weight-connected path, not a trained policy result.

## Execution overlap and shared work

I used two controlled graphs with model and tool delays implemented by `asyncio.sleep`. The 20 seeds share the same graph structure, model batch capacity of eight, eight execution workers, and expansion budget of 128. The barrier and pipeline DAG arms use identical exact merge keys and execution keys. The tree arm has the same concurrency without sharing. The [per-seed receipt](../results/async_execution_20seed.json) records all calls and timings.

| Scenario | Arm | Mean wall | Model expansions | Tool calls | Merges |
|---|---|---:|---:|---:|---:|
| Fast goal behind a slow sibling | Pipeline DAG | 29.0 ms | 3 | 4 | 0 |
| Fast goal behind a slow sibling | Barrier DAG | 73.5 ms | 5 | 5 | 0 |
| Fast goal behind a slow sibling | Pipeline tree | 28.9 ms | 3 | 4 | 0 |
| Exhaustive diamond graph | Pipeline DAG | 68.8 ms | 21 | 30 | 10 |
| Exhaustive diamond graph | Barrier DAG | 80.2 ms | 21 | 30 | 10 |
| Exhaustive diamond graph | Pipeline tree | 124.1 ms | 63 | 62 | 0 |

The first speed gain is from releasing the fast path before its slow sibling finishes. The concurrent tree matches it there; merging contributes nothing in that case. The exhaustive graph shows the separate value of sharing: 21 unique states and 30 tool calls replace 63 path instances and 62 calls. These are controlled timings, not a measured LLM plus real external tool result. Waiting to fill model batches reduced model-call count in this graph but increased latency; the runtime keeps the wait configurable.

I then ran a fixed action graph with real Qwen3-4B GPU forwards on every proposal batch and cancellable `/bin/sleep` tool processes on the H100 host. The model head was executed but its proposed actions were discarded; both arms received the same graph. The task prefix was 193 tokens, with two model batch slots and two execution workers. Over 10 paired seeds, the pipeline solved 10/10 at a median **283.8 ms** versus **383.4 ms** for the barrier. It won all 10 pairs, with a median paired speed ratio of **1.44×** and mean savings of **115 ms**. It averaged 4 tool calls versus 5 for the barrier, while using 3.7 model batches versus 3.0. The [per-seed H100 receipt](../results/native_pipeline_tool_10seed.json) includes every run. The subprocesses deliberately sleep for 8–14 ms except one 100–140 ms branch, so this is an end-to-end overlap measurement under controlled tool latency, not a result on a customer workload or a native action-policy result.

## Native backbone reuse on one H100

The state encoder originally shared token embeddings but did not pass each state through the full 4B backbone. I added two full-backbone paths. `backbone_full` processes the task prefix again with every state batch. `backbone_prefix` fills a prompt KV cache once and runs state suffixes through the same backbone weights. A branch cache copies mutable layer wrappers and expands views of the saved prefix tensors, so it avoids eagerly cloning every GPU KV tensor. The saved prompt cache stayed unchanged in the tests. A configurable `auto` route uses full encoding for small batches and prefix reuse when the prefix token count multiplied by the state batch size reaches 768 on this H100 lane.

These are paired, alternating, warm BF16 measurements on one H100 PCIe using pinned `Qwen/Qwen3-4B-Base` snapshot `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Each row encodes the same fixed canonical states through the full model. Times are medians over eight repetitions, excluding model load and the one-time task prefill. The full [30-token](../results/native_prefix_h100_1x_16state.json), [233-token one-state](../results/native_prefix_h100_8x_1state.json), [233-token four-state](../results/native_prefix_h100_8x_4state.json), [233-token 16-state](../results/native_prefix_h100_8x_16state.json), [929-token one-state](../results/native_prefix_h100_32x_1state.json), and [929-token 16-state](../results/native_prefix_h100_32x_16state.json) receipts include individual repetitions, vector comparisons, and action agreement.

| Shared prefix | States | Full backbone | Shared prefix KV | Result |
|---:|---:|---:|---:|---|
| 30 tokens | 16 | 41.0 ms | 42.2 ms | Similar |
| 233 tokens | 1 | 32.3 ms | 42.6 ms | Full is faster |
| 233 tokens | 4 | 34.2 ms | 28.5 ms | KV is 1.20× faster |
| 233 tokens | 16 | 138.9 ms | 42.4 ms | KV is 3.28× faster |
| 929 tokens | 1 | 32.7 ms | 28.5 ms | KV is 1.15× faster |
| 929 tokens | 16 | 578.3 ms | 48.9 ms | KV is 11.83× faster |

The cached and full state vectors had minimum cosine similarity at least 0.99992 in these rows. For the random frontier head, the first action matched in every tested state, while the complete top-four lists often differed. Small BF16 differences can reorder near-tied actions; a trained head needs its own action and outcome parity evaluation before this route can be treated as behavior-preserving. Peak allocated GPU memory in the combined 929-token, 16-state test was about 10.6 GiB. This is a speedup over full-prefix recomputation, not over an optimized engine with automatic prefix caching. [Hugging Face documents prefix cache reuse](https://huggingface.co/docs/transformers/v4.57.0/kv_cache), and [vLLM already supports prefix caching and continuous batching](https://docs.vllm.ai/en/stable/).

## Current end-to-end boundary

The [H100 native runtime smoke](../results/native_async_h100_auto_smoke.json) loaded the same 4B weights, used a random frontier head, recorded every selected successor prediction and actual execution, and ran both schedulers at an eight-expansion cap. The arithmetic executor is nearly free. The auto route chose full state encoding on every model call because the prompt and batches were small. The four runs did not show a pipeline latency win; three runs were around 0.15 seconds and one pipeline run took 0.34 seconds. Their action trajectories can vary with batch composition, so this smoke is a wiring and cost check, not a fair answer-quality comparison.

The next performance gate is an end-to-end workload with real, variable-latency customer-style executions and the same frozen native policy under a concurrent cached baseline. Measure verified completion latency, throughput, total model FLOPs, tool calls, GPU memory, cancellation cost, and behavior parity at matched budgets. Prefix KV reuse and asynchronous execution are known techniques. A broader speed claim needs a win against an optimized engine that already caches prefixes and batches requests. Training the native reasoner and successor head is a later capability gate.

The local suite passed 50 tests; the nine async/native tests also passed on the H100 host. To reproduce the scheduler receipt, run `.venv/bin/python experiments/async_execution_benchmark.py --seeds 20 --output results/async_execution_20seed.json`. The H100 scripts are [`native_prefix_benchmark.py`](../experiments/native_prefix_benchmark.py), [`native_pipeline_tool_benchmark.py`](../experiments/native_pipeline_tool_benchmark.py), and [`native_async_h100_smoke.py`](../experiments/native_async_h100_smoke.py); pass the pinned model snapshot path shown in each receipt. The GPU was idle after the bounded runs.
