"""Configuration for the self-contained Interference Qwen3 model family."""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class InterferenceQwen3Config(Qwen3Config):
    model_type = "interference_qwen3"

    def __init__(self, frontier_slots=8, frontier_dim=128,
                 frontier_heads=4, interference_scale=1.0,
                 interference_layer_stride=1,
                 tool_error_token_id=None, architecture_version=2, **kwargs):
        super().__init__(**kwargs)
        if frontier_slots < 2:
            raise ValueError("frontier_slots must be at least two")
        if frontier_dim < 1 or frontier_dim % frontier_heads:
            raise ValueError("frontier_dim must be divisible by frontier_heads")
        if interference_layer_stride < 1:
            raise ValueError("interference_layer_stride must be positive")
        self.frontier_slots = frontier_slots
        self.frontier_dim = frontier_dim
        self.frontier_heads = frontier_heads
        self.interference_scale = interference_scale
        self.interference_layer_stride = interference_layer_stride
        self.tool_error_token_id = tool_error_token_id
        self.architecture_version = architecture_version


__all__ = ["InterferenceQwen3Config"]
