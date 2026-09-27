import asyncio

from interference_search.async_execution import AsyncExecutionSearch
from interference_search.native_interference_search import synchronized_search
from interference_search.mbpp_async_domain import (
    CodeState, MBPPAsyncDomain, canonical_source, extract_code,
)
from interference_search.sandboxed_python import SandboxResult
from interference_search.vllm_code_backend import CodeGeneration


class FakeBackend:
    def format_prompt(self, instruction):
        return instruction

    async def generate_batch(self, prompts, seeds):
        return [CodeGeneration(
            "```python\ndef f(x):\n    return x + 1\n```"
            if "general algorithm" in prompt or "Fix this" in prompt
            else "def f(x): return x",
            len(prompt), 10, 0.0,
        ) for prompt in prompts]

    async def generate_stream(self, prompts, seeds):
        for index, result in enumerate(await self.generate_batch(prompts, seeds)):
            yield index, result


class FakeSandbox:
    async def run(self, source, tests, imports):
        passes = "x + 1" in source
        return SandboxResult((("pass", "") if passes else
                              ("fail", "AssertionError: got 1; expected 2"),), 0.001)


PROBLEM = {"task_id": 1, "prompt": "Write f(x) that returns x + 1.",
           "test_list": ["assert f(1) == 2"], "test_imports": []}


def test_extract_code_and_exact_ast_key():
    assert extract_code("Here:\n```python\ndef f(x): return x + 1\n```") == (
        "def f(x): return x + 1")
    assert canonical_source("def f(x): return x+1") == canonical_source(
        "def f( x ):\n  return x + 1")
    outcome = SandboxResult((("fail", "AssertionError"),), 0.001)
    domain = MBPPAsyncDomain(FakeBackend(), FakeSandbox())
    assert domain.key(CodeState("def f(x): return x", outcome, 1)) != domain.key(
        CodeState("def f(x): return x + 2", outcome, 1))
    assert domain.key(CodeState("def f(x): return x", outcome, 1)) != domain.key(
        CodeState("def f(x): return x", outcome, 2))


def test_same_backend_and_executor_solve_in_both_schedulers():
    async def exercise(mode):
        domain = MBPPAsyncDomain(FakeBackend(), FakeSandbox())
        return await AsyncExecutionSearch(model_batch_size=2, execution_workers=2,
                                          mode=mode).run(domain, PROBLEM, 4)

    pipeline = asyncio.run(exercise("pipeline"))
    barrier = asyncio.run(exercise("barrier"))
    assert pipeline.solved and barrier.solved
    assert pipeline.execution_requests == barrier.execution_requests == 2
    assert "x + 1" in pipeline.solution.source


def test_synchronized_code_frontier_executes_and_uses_verifier_scores():
    domain = MBPPAsyncDomain(FakeBackend(), FakeSandbox(), fresh_candidates=2,
                             revision_candidates=1, max_depth=2)
    result = asyncio.run(synchronized_search(
        domain, PROBLEM, expansion_budget=3, width=1,
        execution_workers=2, merge=True))
    assert result.solved
    assert result.executions == 2
    assert result.events[0]["goal"]["child"].outcome.all_passed


def test_pruned_code_is_shared_as_a_refutation():
    domain = MBPPAsyncDomain(FakeBackend(), FakeSandbox())
    context = domain.prepare(PROBLEM)
    failed = CodeState("def f(x): return x", SandboxResult(
        (("fail", "AssertionError: got 1; expected 2"),), 0.001), 1)
    domain.observe_pruned([failed], context)
    prompt = domain._prompt(CodeState(None, None), context, 0)
    assert "Shared refutations from sibling branches" in prompt
    assert "def f(x): return x" in prompt
    assert domain.score_states([failed], context) == [-0.02]
