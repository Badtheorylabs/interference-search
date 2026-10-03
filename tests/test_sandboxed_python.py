import asyncio
import shutil

import pytest

from interference_search.sandboxed_python import BubblewrapPython


pytestmark = pytest.mark.skipif(
    not shutil.which("bwrap") or not shutil.which("prlimit"),
    reason="Linux bubblewrap and prlimit required",
)


def test_candidate_runs_tests_without_host_home_or_network():
    async def exercise():
        runner = BubblewrapPython()
        good = await runner.run("def f(x): return x + 1",
                                ["assert f(1) == 2", "assert f(2) == 3"])
        wrong = await runner.run("def f(x): return x", ["assert f(1) == 2"])
        home = await runner.run("open('/home/ubuntu/.ssh/id_ed25519_new').read()",
                                ["assert True"])
        network = await runner.run(
            "import socket; socket.socket().connect(('1.1.1.1', 80))",
            ["assert True"],
        )
        return good, wrong, home, network

    good, wrong, home, network = asyncio.run(exercise())
    assert good.all_passed and good.passed == 2
    assert not wrong.all_passed and wrong.tests[0][0] == "fail"
    assert "got 1; expected 2" in wrong.tests[0][1]
    assert home.tests[0][0] == "load_error"
    assert "No such file" in home.tests[0][1]
    assert network.tests[0][0] == "load_error"
    assert "Network is unreachable" in network.tests[0][1]


def test_eight_candidates_can_execute_concurrently():
    async def exercise():
        runner = BubblewrapPython()
        return await asyncio.gather(*(
            runner.run("def f(x): return x + 1", ["assert f(2) == 3"])
            for _ in range(8)
        ))

    assert all(result.all_passed for result in asyncio.run(exercise()))


def test_child_cpu_limit_stops_a_candidate_that_ignores_alarms():
    source = "import signal; signal.signal(signal.SIGALRM, lambda *_: None)\nwhile True: pass"
    result = asyncio.run(BubblewrapPython(wall_seconds=6).run(source, ["assert True"]))
    assert not result.all_passed
    assert result.wall_seconds < 6


def test_cancel_kills_a_running_sandbox_process():
    async def exercise():
        runner = BubblewrapPython()
        task = asyncio.create_task(runner.run("import time; time.sleep(5)",
                                              ["assert True"]))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(exercise(), timeout=2))


def test_abrupt_candidate_exit_is_a_failed_program_not_infrastructure_failure():
    result = asyncio.run(BubblewrapPython().run("import os; os._exit(137)",
                                                  ["assert True"]))
    assert result.tests[0][0] == "candidate_crash"
    assert result.sandbox_error is None
