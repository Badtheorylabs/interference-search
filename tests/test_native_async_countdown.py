import asyncio
from types import SimpleNamespace

import torch
from torch import nn

from interference_search.async_execution import AsyncExecutionSearch
from interference_search.native_async_countdown import NativeAsyncCountdownDomain
from interference_search.native_lm import NativeFrontierLM


class TinyBody(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.projection = nn.Linear(32, 32)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.projection(self.embeddings(input_ids)))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embeddings = nn.Embedding(128, 32)
        self.model = TinyBody(self.embeddings)

    def get_input_embeddings(self):
        return self.embeddings


class TinyTokenizer:
    def encode(self, text):
        return [1 + ord(character) % 127 for character in text]


def test_native_weights_feed_async_search_with_one_task_encoding():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = NativeFrontierLM(TinyBackbone(), actions=48, heads=4)
    domain = NativeAsyncCountdownDomain(model, TinyTokenizer(), max_numbers=4,
                                        actions_per_state=2, state_mode="lightweight")
    counts = {"task": 0}
    original = model.encode_task

    def counted(*args, **kwargs):
        counts["task"] += 1
        return original(*args, **kwargs)

    model.encode_task = counted
    problem = {"numbers": [2, 3, 4, 5], "target": 19}

    async def exercise():
        context = await domain.prepare(problem)
        start = domain.start(problem)
        first = await domain.propose_batch([start], context)
        second = await domain.propose_batch([start], context)
        assert [item.action for item in first[0]] == [item.action for item in second[0]]
        assert len(context.state_vectors) == 1
        assert all(domain.codec.legal(start, item.action) for item in first[0])
        assert all(item.predicted_successor.shape == (32,) for item in first[0])
        domain.observe_execution(start, first[0][0].action, (999,), context)
        assert context.refuted_states == [(999,)]
        return await AsyncExecutionSearch(model_batch_size=2, execution_workers=2).run(
            domain, problem, expansion_budget=3)

    result = asyncio.run(exercise())
    assert result.prepare_calls == 1
    assert result.expansions == 3
    assert result.execution_calls > 0
    assert result.observations == result.execution_calls
    assert result.predictions_recorded >= result.observations
    assert counts["task"] == 2  # one explicit preparation and one search preparation


def test_prefix_cache_fork_shares_immutable_tensors_without_mutating_base():
    class Layer:
        def __init__(self):
            self.keys = torch.randn(1, 2, 3, 4)
            self.values = torch.randn(1, 2, 3, 4)

    class Cache:
        def __init__(self):
            self.layers = [Layer()]

    base = Cache()
    before = base.layers[0].keys.clone()
    branch = NativeAsyncCountdownDomain._fork_prompt_cache(base, 4)
    assert branch.layers[0] is not base.layers[0]
    assert branch.layers[0].keys.shape == (4, 2, 3, 4)
    assert branch.layers[0].keys.data_ptr() == base.layers[0].keys.data_ptr()
    branch.layers[0].keys = torch.cat((branch.layers[0].keys,
                                      torch.zeros(4, 2, 1, 4)), dim=-2)
    assert base.layers[0].keys.shape == (1, 2, 3, 4)
    torch.testing.assert_close(base.layers[0].keys, before)


def test_auto_encoder_route_uses_calibrated_prompt_batch_threshold():
    model = NativeFrontierLM(TinyBackbone(), actions=48, heads=4)
    domain = NativeAsyncCountdownDomain(model, TinyTokenizer(), max_numbers=4,
                                        state_mode="auto", prefix_cache_min_work=768)
    assert domain._select_encoder_mode(30, 16) == "backbone_full"
    assert domain._select_encoder_mode(233, 1) == "backbone_full"
    assert domain._select_encoder_mode(233, 4) == "backbone_prefix"
    assert domain._select_encoder_mode(929, 1) == "backbone_prefix"
