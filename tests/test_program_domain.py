from interference_search.program_domain import Code, State, execute, extract_code, feedback
from interference_search.frontier_v2 import Candidate, select_frontier

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


def test_matching_visible_tests_do_not_prove_program_equivalence():
    one = State("def add(a, b):\n    return a + b", execute("def add(a, b):\n    return a + b", TESTS))
    two = State("def add(a, b):\n    return 3 if a == 1 else 4", execute(
        "def add(a, b):\n    return 3 if a == 1 else 4", TESTS))
    formatted = State("def add(a,b):\n  return a+b", one.behaviour)
    domain = Code.__new__(Code)
    assert domain.signature(one) == domain.signature(two)
    assert domain.exact_key(one) != domain.exact_key(two)
    assert domain.exact_key(one) == domain.exact_key(formatted)
    assert execute(two.src, ["assert add(8, 9) == 17"])[0][0] == "fail"


def test_new_frontier_preserves_alternate_programs_and_signature_diversity():
    candidates = [
        Candidate("a", "a", "same", 2.0),
        Candidate("b", "b", "same", 2.0),
        Candidate("c", "c", "other", 1.0),
        Candidate("a-repeat", "a", "same", 2.0),
    ]
    selected = select_frontier(candidates, width=2)
    assert {candidate.exact_key for candidate in selected.live} == {"a", "c"}
    assert selected.exact_duplicates == 1
    assert selected.signature_collisions == 1
    assert select_frontier(candidates, width=3).live[2].exact_key == "b"


def test_errors_and_timeouts_are_contained():
    assert execute("def add(a, b):\n    raise ValueError('no')", TESTS)[0][0] == "error"
    assert execute("while True:\n    pass", TESTS)[0][0] in ("load", "error")


def test_extract_code_and_feedback():
    src = extract_code("Here:\n```python\ndef add(a, b):\n    return a + b\n```")
    assert src.startswith("def add")
    s = State(src, execute(src, TESTS))
    assert feedback({"test_list": TESTS}, s).count("PASS") == 2
