"""Bounded model-generation smoke for the pinned MBPP vLLM backend."""

import argparse
import asyncio
import json

from interference_search.vllm_code_backend import VLLMCodeBackend


async def run(args):
    backend = VLLMCodeBackend(args.model, max_tokens=96,
                              enforce_eager=args.enforce_eager)
    try:
        instruction = ("Write a Python function named add_two that returns x + 2. "
                       "Reply with only code in one Python code block.")
        prompt = backend.format_prompt(instruction)
        output = await backend.generate_one(prompt)
        result = {"model": args.model, "enforce_eager": args.enforce_eager,
                  "prompt_tokens": output.prompt_tokens,
                  "output_tokens": output.output_tokens,
                  "generation_seconds": round(output.wall_seconds, 3),
                  "text": output.text}
        print(json.dumps(result, indent=2))
    finally:
        await backend.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
