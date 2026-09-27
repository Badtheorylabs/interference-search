"""MBPP code proposals, sandbox execution, and exact repeat keys.

Both scheduler arms use one deterministic model backend and this domain. A
visible test appears in the initial prompt; execution returns feedback from
the frozen full test list. Exact AST plus depth identifies revisable states.
Matching finite-test behavior alone is never used as a hard merge key.
"""

import ast
import hashlib
import re
from dataclasses import dataclass

from .async_execution import AsyncAction
from .sandboxed_python import BubblewrapPython, SandboxResult


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    if blocks:
        return blocks[0].strip()
    match = re.search(r"(?:^|\n)(def |import |from )", text)
    return text[match.start(1):].strip() if match else text.strip()


def canonical_source(source: str):
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        return source


@dataclass(frozen=True)
class CodeState:
    source: str | None
    outcome: SandboxResult | None
    depth: int = 0


class MBPPAsyncDomain:
    def __init__(self, backend, sandbox: BubblewrapPython,
                 fresh_candidates: int = 2, revision_candidates: int = 1,
                 max_depth: int = 3):
        self.backend = backend
        self.sandbox = sandbox
        self.fresh_candidates = fresh_candidates
        self.revision_candidates = revision_candidates
        self.max_depth = max_depth

    def start(self, problem):
        return CodeState(None, None)

    def key(self, state: CodeState):
        if state.source is None:
            return ("root",)
        return (state.depth, canonical_source(state.source),
                tuple(state.outcome.tests) if state.outcome else ())

    def _prompt(self, state: CodeState, problem, variant: int) -> str:
        if state.source is None:
            styles = (
                "Use a clear, direct implementation.",
                "Use a general algorithm that handles edge cases.",
                "Use a concise implementation with standard Python libraries.",
            )
            instruction = (
                f"Write the Python function described here:\n{problem['prompt']}\n"
                f"It must pass this example:\n{problem['test_list'][0]}\n"
                f"{styles[variant % len(styles)]}\n"
                "Return only Python source code, with no explanation."
            )
        else:
            feedback = "\n".join(
                f"{status.upper()} {test}: {detail}" for test, (status, detail)
                in zip(problem["test_list"], state.outcome.tests)
            )
            instruction = (
                f"Fix this Python function for the task:\n{problem['prompt']}\n"
                f"Current code:\n```python\n{state.source[:5000]}\n```\n"
                f"Test feedback:\n{feedback[:5000]}\n"
                "Return only corrected Python source code, with no explanation."
            )
        return self.backend.format_prompt(instruction)

    def _requests(self, states: list[CodeState], problem):
        prompts = []
        seeds = []
        owners = []
        for index, state in enumerate(states):
            if state.depth >= self.max_depth:
                continue
            variants = self.fresh_candidates if state.source is None else self.revision_candidates
            for variant in range(variants):
                prompt = self._prompt(state, problem, variant)
                prompts.append(prompt)
                seed_text = f"{problem['task_id']}:{state.depth}:{variant}:{state.source or ''}"
                seeds.append(int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))
                owners.append(index)
        return prompts, seeds, owners

    async def propose_batch(self, states: list[CodeState], problem):
        prompts, seeds, owners = self._requests(states, problem)
        results = await self.backend.generate_batch(prompts, seeds) if prompts else []
        batches = [[] for _ in states]
        for owner, generated in zip(owners, results):
            source = extract_code(generated.text)
            batches[owner].append(AsyncAction(source))
        return batches

    async def propose_stream(self, states: list[CodeState], problem):
        prompts, seeds, owners = self._requests(states, problem)
        async for request_index, generated in self.backend.generate_stream(prompts, seeds):
            yield owners[request_index], [AsyncAction(extract_code(generated.text))]

    async def execute(self, state: CodeState, action: str, problem):
        outcome = await self.sandbox.run(action, problem["test_list"],
                                         problem.get("test_imports", []))
        return CodeState(action, outcome, state.depth + 1)

    def execution_key(self, state: CodeState, action: str, problem):
        return (problem["task_id"], state.depth, canonical_source(action))

    def is_goal(self, state: CodeState, problem):
        return state.outcome is not None and state.outcome.all_passed

    def certified_dead(self, state: CodeState, problem):
        return False
