"""Bounded MBPP Python execution in an isolated bubblewrap process.

Generated source and benchmark tests enter through stdin. The sandbox has
read-only system Python, a temporary filesystem, no host home directory, and
no network. Parent-side wall limits, process-group cleanup, and child-side
resource limits apply to every candidate.
"""

import asyncio
import json
import os
import signal
import tempfile
import time
from dataclasses import dataclass


RUNNER = r'''
import ast, contextlib, io, json, signal, sys

def alarm(*args):
    raise TimeoutError("per-test timeout")

signal.signal(signal.SIGALRM, alarm)
payload = json.load(sys.stdin)
source = payload["source"]
tests = payload["tests"]
imports = payload["imports"]
namespace = {}
results = []
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        signal.alarm(2)
        for statement in imports:
            exec(compile(statement, "<import>", "exec"), namespace)
        exec(compile(source, "<candidate>", "exec"), namespace)
        signal.alarm(0)
except BaseException as error:
    signal.alarm(0)
    results = [["load_error", type(error).__name__ + ": " + str(error)[:160]]] * len(tests)
else:
    for test in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                signal.alarm(2)
                tree = ast.parse(test)
                assertion = tree.body[0] if len(tree.body) == 1 else None
                comparison = assertion.test if isinstance(assertion, ast.Assert) else None
                if (isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
                        and isinstance(comparison.ops[0], ast.Eq)):
                    actual = eval(compile(ast.Expression(comparison.left),
                                          "<actual>", "eval"), namespace)
                    expected = eval(compile(ast.Expression(comparison.comparators[0]),
                                            "<expected>", "eval"), namespace)
                    if actual != expected:
                        raise AssertionError("got " + repr(actual)[:80] +
                                             "; expected " + repr(expected)[:80])
                else:
                    exec(compile(test, "<test>", "exec"), namespace)
                signal.alarm(0)
            results.append(["pass", ""])
        except BaseException as error:
            signal.alarm(0)
            results.append(["fail", type(error).__name__ + ": " + str(error)[:160]])
print(json.dumps(results))
'''


@dataclass(frozen=True)
class SandboxResult:
    tests: tuple[tuple[str, str], ...]
    wall_seconds: float
    sandbox_error: str | None = None

    @property
    def passed(self) -> int:
        return sum(status == "pass" for status, _ in self.tests)

    @property
    def all_passed(self) -> bool:
        return bool(self.tests) and self.passed == len(self.tests)


class BubblewrapPython:
    def __init__(self, wall_seconds: float = 6.0, python_path: str = "/usr/bin/python3"):
        if wall_seconds <= 0:
            raise ValueError("wall limit must be positive")
        self.wall_seconds = wall_seconds
        self.python_path = python_path

    def _command(self):
        return [
            "/usr/bin/sudo", "-n", "-u", "nobody", "--",
            "/usr/bin/prlimit", "--cpu=3", "--as=536870912", "--fsize=131072",
            "--nofile=64", "--nproc=64", "--",
            "/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--dev", "/dev",
            "--proc", "/proc", "--tmpfs", "/tmp", "--dir", "/home",
            "--chdir", "/tmp", "--", self.python_path, "-I", "-S", "-c", RUNNER,
        ]

    @staticmethod
    def _kill_group(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def run(self, source: str, tests: list[str],
                  imports: list[str] | None = None) -> SandboxResult:
        imports = imports or []
        if not isinstance(source, str) or len(source) > 50_000:
            raise ValueError("candidate source must be at most 50,000 characters")
        if not tests or len(tests) > 32 or any(len(test) > 10_000 for test in tests):
            raise ValueError("invalid benchmark test list")
        if any(statement != "import math" for statement in imports):
            raise ValueError("unsupported benchmark import")
        payload = json.dumps({"source": source, "tests": tests,
                              "imports": imports}).encode()
        started = time.perf_counter()
        environment = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0",
                       "LC_ALL": "C.UTF-8"}
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = await asyncio.create_subprocess_exec(
                *self._command(), stdin=asyncio.subprocess.PIPE,
                stdout=stdout, stderr=stderr, env=environment,
                start_new_session=True,
            )
            timed_out = False
            try:
                await asyncio.wait_for(process.communicate(payload),
                                       timeout=self.wall_seconds)
            except asyncio.TimeoutError:
                timed_out = True
                self._kill_group(process)
                await process.wait()
            except asyncio.CancelledError:
                self._kill_group(process)
                await process.wait()
                raise
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read(131_072).decode(errors="replace").strip()
            error = stderr.read(4096).decode(errors="replace").strip()
        elapsed = time.perf_counter() - started
        if timed_out:
            return SandboxResult(tuple(("timeout", "wall limit") for _ in tests),
                                 elapsed)
        try:
            rows = json.loads(output.splitlines()[-1])
            if len(rows) != len(tests) or any(len(row) != 2 for row in rows):
                raise ValueError("wrong result shape")
            return SandboxResult(tuple((str(row[0]), str(row[1])) for row in rows), elapsed,
                                 error or None if process.returncode else None)
        except (IndexError, ValueError, TypeError):
            message = (error or f"runner exited {process.returncode}")[:200]
            infrastructure = message.startswith(("bwrap:", "sudo:", "prlimit:"))
            status = "infrastructure_error" if infrastructure else "candidate_crash"
            return SandboxResult(tuple((status, message) for _ in tests),
                                 elapsed, message if infrastructure else None)
