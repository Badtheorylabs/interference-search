"""One pinned vLLM engine for batched, prefix-cached MBPP proposals."""

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeGeneration:
    text: str
    prompt_tokens: int
    output_tokens: int
    wall_seconds: float


class VLLMCodeBackend:
    def __init__(self, model_path: str, max_tokens: int = 160,
                 enforce_eager: bool = True, gpu_memory_utilization: float = 0.45):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.engine_args = AsyncEngineArgs(
            model=model_path, tokenizer=model_path, dtype="bfloat16",
            max_model_len=4096, max_num_seqs=16,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True, enforce_eager=enforce_eager,
            disable_log_stats=True, trust_remote_code=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.requests = 0
        self.started_requests = 0
        self.started_prompt_tokens = 0
        self.cancelled_requests = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.request_wall_seconds = 0.0
        self.prompt_observations: dict[str, str] = {}
        self.parity_mismatches = 0

    def format_prompt(self, instruction: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    async def generate_one(self, prompt: str, seed: int = 0) -> CodeGeneration:
        from vllm import SamplingParams

        request_id = str(uuid.uuid4())
        params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens, seed=seed)
        started = time.perf_counter()
        self.started_requests += 1
        self.started_prompt_tokens += len(self.tokenizer.encode(
            prompt, add_special_tokens=False))
        final = None
        try:
            async for output in self.engine.generate(prompt, params, request_id):
                final = output
        except asyncio.CancelledError:
            self.cancelled_requests += 1
            if final is not None and final.outputs:
                self.output_tokens += len(final.outputs[0].token_ids or ())
            raise
        finally:
            self.request_wall_seconds += time.perf_counter() - started
        if final is None or not final.outputs:
            raise RuntimeError("vLLM returned no completion")
        generation = CodeGeneration(
            final.outputs[0].text,
            len(final.prompt_token_ids or ()),
            len(final.outputs[0].token_ids or ()),
            time.perf_counter() - started,
        )
        self.requests += 1
        self.prompt_tokens += generation.prompt_tokens
        self.output_tokens += generation.output_tokens
        prior = self.prompt_observations.setdefault(prompt, generation.text)
        if prior != generation.text:
            self.parity_mismatches += 1
        return generation

    async def generate_batch(self, prompts: list[str], seeds: list[int]):
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must align")
        return await asyncio.gather(*(
            self.generate_one(prompt, seed) for prompt, seed in zip(prompts, seeds)
        ))

    async def generate_stream(self, prompts: list[str], seeds: list[int]):
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must align")

        async def indexed(index, prompt, seed):
            return index, await self.generate_one(prompt, seed)

        tasks = [asyncio.create_task(indexed(index, prompt, seed))
                 for index, (prompt, seed) in enumerate(zip(prompts, seeds))]
        try:
            for completed in asyncio.as_completed(tasks):
                yield await completed
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self):
        value = self.engine.shutdown()
        if inspect.isawaitable(value):
            await value
