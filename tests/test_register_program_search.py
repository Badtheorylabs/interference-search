import itertools

from interference_search.register_program import RegisterProgram
from interference_search.register_program_search import CachedFactorSolver, CompactQuotientSolver


def test_factor_and_compact_candidates_match_after_same_feedback():
    for topology in ("independent", "ring", "sparse"):
        task = RegisterProgram(registers=3, slots_per_register=2, topology=topology, seed=7)
        factor = CachedFactorSolver(task.public_view())
        compact = CompactQuotientSolver(task.public_view())
        for _ in range(task.registers + 2):
            factor_batch = factor.batch(4)
            compact_batch = compact.batch(4)
            assert factor_batch == compact_batch
            if not factor_batch:
                break
            feedback = [task.verify(candidate) for candidate in factor_batch]
            if None in feedback:
                break
            for item in feedback:
                factor.update(item)
                compact.update(item)
            survivors = [choices for choices in itertools.product((0, 1), repeat=task.decisions)
                         if all(task.local_output(register,
                            tuple(choices[index] for index in task.scope(register)))
                            == task.expected[register]
                            for register in factor.factors)]
            assert compact.state.count() == len(survivors)
            assert compact.state.contains(task.hidden)


def test_feedback_exclusions_are_certified_and_batch_is_unique():
    task = RegisterProgram(registers=3, slots_per_register=2, topology="ring", seed=9)
    solver = CompactQuotientSolver(task.public_view())
    batch = solver.batch(8)
    assert len(batch) == len(set(batch)) == 8
    assert solver.state.count() == 1 << task.decisions
    feedback = task.verify(batch[0])
    assert feedback is not None
    solver.update(feedback)
    assert solver.state.contains(task.hidden)
    assert all(any(task.hidden[index] != bit for index, bit in ref.bindings)
               for ref in solver.state.refutations)
