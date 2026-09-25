import random

from interference_search.countdown import (apply_move, check_answer, children, extract_attempts, gen_hard_problem,
                                           moves, parse_expr, reachable_states, solver)


def test_moves_are_legal_and_sorted():
    for s in moves((3, 25, 50, 7)):
        assert len(s) == 3 and list(s) == sorted(s) and all(x > 0 for x in s)


def test_children_merge_duplicates():
    raw, merged = moves((2, 2, 4)), children((2, 2, 4))
    assert len(merged) == len(set(raw)) < len(raw)


def test_apply_move():
    assert apply_move((3, 7, 25, 50), 25, "*", 3) == (7, 50, 75)
    assert apply_move((3, 7), 7, "/", 3) is None        # not a whole number
    assert apply_move((3, 7), 3, "-", 7) is None        # not positive


def test_solver_matches_a_known_solution():
    solvable = solver(361)
    assert solvable((3, 19, 24, 98))                    # ((24 * 19) + 3) - 98
    assert solvable((3, 98, 456))
    assert not solvable((355,))


def test_canonical_forms_ignore_order():
    a, b = parse_expr("(25 + 3) * 7"), parse_expr("7 * (3 + 25)")
    assert a[0] == b[0] == 196 and a[1] == b[1]


def test_extract_attempts_rebuilds_step_chains():
    text = "25 * 3 = 75, then 75 + 50 = 125, then 125 + 7 = 132. <answer>50 + 25 * 3 + 7</answer>"
    atts = extract_attempts(text, [25, 7, 3, 50], 132)
    assert any(a["correct"] for a in atts)
    assert check_answer(text, [25, 7, 3, 50], 132)


def test_word_operators_and_offsets():
    text = "20 times 11 is 220. 30 times 7 is 210. Maybe 220 plus 210 is 430."
    atts = extract_attempts(text, [20, 11, 7, 30], 390)
    assert [a["value"] for a in atts] == [430.0]
    assert text[atts[0]["offset"]:].lstrip().startswith("220")


def test_hard_problems_are_solvable():
    rng = random.Random(0)
    for _ in range(5):
        p = gen_hard_problem(rng)
        assert solver(p["target"])(tuple(sorted(p["numbers"])))
        assert len(reachable_states(p["numbers"])) > 10
