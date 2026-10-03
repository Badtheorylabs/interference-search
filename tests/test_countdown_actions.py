import random

from interference_search.countdown import children
from interference_search.countdown_actions import CountdownActionCodec


def test_model_action_slots_cover_exactly_the_legal_countdown_children():
    codec = CountdownActionCodec(max_numbers=7)
    rng = random.Random(19)
    for count in range(2, 8):
        for _ in range(15):
            state = tuple(sorted(rng.randint(1, 30) for _ in range(count)))
            legal = [codec.apply(state, k) for k in range(len(codec.actions))
                     if codec.legal(state, k)]
            assert set(legal) == set(children(state))
            assert all(child is not None and len(child) == count - 1 for child in legal)


def test_padded_and_illegal_actions_are_not_executed():
    codec = CountdownActionCodec(max_numbers=7)
    state = (3, 7)
    assert len(codec.legal_mask(state)) == 168
    for k, action in enumerate(codec.actions):
        if action.left >= 2 or action.right >= 2:
            assert not codec.legal(state, k)
            assert codec.apply(state, k) is None
