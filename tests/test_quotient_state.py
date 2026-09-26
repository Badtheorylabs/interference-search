import itertools
import random

import pytest
import torch

from interference_search.quotient_state import QuotientState, VerifiedRefutation


def test_compact_state_matches_brute_force_after_verified_updates():
    rng = random.Random(61)
    for decisions in (4, 7, 9):
        state = QuotientState(decisions)
        hidden = tuple(rng.randrange(2) for _ in range(decisions))
        excluded = []
        for _ in range(12):
            positions = rng.sample(range(decisions), rng.randint(1, min(4, decisions)))
            values = {index: rng.randrange(2) for index in positions}
            if all(hidden[index] == bit for index, bit in values.items()):
                values[positions[0]] = 1 - hidden[positions[0]]
            certificate = VerifiedRefutation.from_mapping(values)
            assert state.refute(certificate, lambda item: any(
                hidden[index] != bit for index, bit in item.bindings
            )) in (True, False)
            excluded.append(certificate)
            surviving = [bits for bits in itertools.product((0, 1), repeat=decisions)
                         if all(not all(bits[index] == bit for index, bit in ref.bindings)
                                for ref in excluded)]
            assert state.count() == len(surviving)
            assert state.contains(hidden)
            assert state.recover() in surviving


def test_one_refutation_removes_unvisited_region_and_recovery_survives():
    state = QuotientState(32)
    assert state.count() == 2**32
    refutation = VerifiedRefutation.from_mapping({1: 1, 4: 0, 8: 1, 12: 0, 23: 1})
    assert state.refute(refutation, lambda _: True)
    assert state.count() == 2**32 - 2**27
    assert state.reachable_nodes() < 40
    first = state.recover()
    assert first is not None
    assert state.refute(VerifiedRefutation.from_mapping(dict(enumerate(first))), lambda _: True)
    assert state.count() == 2**32 - 2**27 - 1
    assert state.recover() != first


def test_uncertified_or_empty_refutation_cannot_erase_the_state():
    state = QuotientState(6)
    with pytest.raises(ValueError, match="verifier"):
        state.refute(VerifiedRefutation.from_mapping({0: 1}), lambda _: False)
    with pytest.raises(ValueError, match="empty"):
        state.refute(VerifiedRefutation.from_mapping({}), lambda _: True)
    assert state.count() == 64


def test_neural_weights_and_gradients_match_explicit_small_version_space():
    state = QuotientState(5)
    state.refute(VerifiedRefutation.from_mapping({0: 1, 2: 0}), lambda _: True)
    state.refute(VerifiedRefutation.from_mapping({1: 0, 4: 1}), lambda _: True)
    logits = torch.tensor([0.3, -0.7, 1.1, 0.2, -0.4], requires_grad=True)
    compact = state.log_partition(logits)
    assignments = [bits for bits in itertools.product((0, 1), repeat=5) if state.contains(bits)]
    scores = torch.stack([sum(logits[i] * bit for i, bit in enumerate(bits))
                          for bits in assignments])
    explicit = torch.logsumexp(scores, dim=0)
    torch.testing.assert_close(compact, explicit)
    compact.backward()
    probabilities = scores.detach().softmax(0)
    expected_marginals = torch.tensor([
        sum(float(weight) * bits[index] for weight, bits in zip(probabilities, assignments))
        for index in range(5)
    ])
    torch.testing.assert_close(logits.grad, expected_marginals, atol=1e-6, rtol=1e-6)
    best = state.best(logits)
    assert best in assignments
    frozen_logits = logits.detach().tolist()
    assert sum(frozen_logits[i] * bit for i, bit in enumerate(best)) == pytest.approx(
        max(sum(frozen_logits[i] * bit for i, bit in enumerate(bits)) for bits in assignments)
    )
