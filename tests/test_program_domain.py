from interference_search.program_domain import State, execute, extract_code, feedback

TESTS = ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]


def test_passing_program():
    b = execute("def add(a, b):\n    return a + b", TESTS)
    assert [x[0] for x in b] == ["pass", "pass"]


def test_failing_program_records_what_it_returned():
    b = execute("def add(a, b):\n    return a * b", TESTS)
    assert b[0] == ("fail", "2") and b[1][0] == "pass"


def test_equal_behaviour_gives_equal_keys():
    one = State("def add(a, b):\n    return a * b", execute("def add(a, b):\n    return a * b", TESTS))
    two = State("def add(x, y):\n    return y * x", execute("def add(x, y):\n    return y * x", TESTS))
    assert tuple(one.behaviour) == tuple(two.behaviour)


def test_errors_and_timeouts_are_contained():
    assert execute("def add(a, b):\n    raise ValueError('no')", TESTS)[0][0] == "error"
    assert execute("while True:\n    pass", TESTS)[0][0] in ("load", "error")


def test_extract_code_and_feedback():
    src = extract_code("Here:\n```python\ndef add(a, b):\n    return a + b\n```")
    assert src.startswith("def add")
    s = State(src, execute(src, TESTS))
    assert feedback({"test_list": TESTS}, s).count("PASS") == 2
