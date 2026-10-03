import torch

from interference_search.native_kv_frontier import NativeKVFrontier


def test_kv_fork_shares_prefix_storage_but_not_mutable_layer_wrapper():
    class Layer:
        def __init__(self):
            self.keys = torch.randn(1, 2, 3, 4)
            self.values = torch.randn(1, 2, 3, 4)

    class Cache:
        def __init__(self):
            self.layers = [Layer(), Layer()]

    base = Cache()
    before = base.layers[0].keys.clone()
    branch = NativeKVFrontier.fork_prompt_cache(base, 5)
    assert branch is not base
    assert branch.layers[0] is not base.layers[0]
    assert branch.layers[0].keys.shape == (5, 2, 3, 4)
    assert branch.layers[0].keys.data_ptr() == base.layers[0].keys.data_ptr()
    branch.layers[0].keys = torch.cat(
        (branch.layers[0].keys, torch.zeros(5, 2, 1, 4)), dim=-2)
    assert base.layers[0].keys.shape == (1, 2, 3, 4)
    torch.testing.assert_close(base.layers[0].keys, before)
