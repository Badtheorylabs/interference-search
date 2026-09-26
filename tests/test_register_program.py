import random

from interference_search.register_program import RegisterProgram


def test_local_certificate_matches_full_program_execution():
    for topology in ("independent", "ring", "sparse"):
        task = RegisterProgram(topology=topology, seed=17)
        assert task.verify(task.hidden) is None
        rng = random.Random(91)
        for _ in range(40):
            choices = tuple(rng.randrange(2) for _ in range(task.decisions))
            outputs = task.execute(choices)
            for register in range(task.registers):
                scope_values = tuple(choices[index] for index in task.scope(register))
                assert task.local_output(register, scope_values) == outputs[register]
            failure = task.verify(choices)
            if failure is not None:
                scope_values = tuple(choices[index] for index in failure.scope)
                assert task.certify_forbidden(failure, scope_values)
                assert scope_values not in task.allowed_patterns(failure)


def test_dependency_topologies_change_factor_width():
    independent = RegisterProgram(topology="independent", seed=2)
    ring = RegisterProgram(topology="ring", seed=2)
    assert all(len(independent.scope(register)) == 4 for register in range(6))
    assert all(len(ring.scope(register)) == 8 for register in range(6))


def test_public_grammar_excludes_hidden_target():
    task = RegisterProgram(seed=4)
    grammar = task.public_view()
    assert not hasattr(grammar, "hidden")
    assert not hasattr(grammar, "expected")
    assert grammar.scope(0) == task.scope(0)
    feedback = task.verify(tuple(0 for _ in range(task.decisions)))
    assert feedback is not None
    assert grammar.allowed_patterns(feedback) == task.allowed_patterns(feedback)
