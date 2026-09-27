"""Native frontier attached to a decoder backbone and its live prompt KV."""

import copy
import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from .native_frontier import NativeFrontierCell


@dataclass
class NativeKVContext:
    task_vector: torch.Tensor
    prompt_cache: object
    prefix_tokens: int


class NativeKVFrontier(nn.Module):
    """Branch frontier states from one task KV and consume model hidden states."""

    def __init__(self, backbone: nn.Module, actions: int, heads: int = 8,
                 scale: float = 0.5, train_backbone: bool = False,
                 match_space: str = "model"):
        super().__init__()
        self.backbone = backbone
        self.train_backbone = train_backbone
        self.match_space = match_space
        if not train_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.cell = NativeFrontierCell(backbone.config.hidden_size, actions,
                                       heads=heads, scale=scale)
        self.task_prefills = 0
        self.suffix_forwards = 0

    @staticmethod
    def fork_prompt_cache(cache, batch_size: int):
        if isinstance(cache, (tuple, list)):
            return type(cache)(
                (keys.expand(batch_size, *keys.shape[1:]),
                 values.expand(batch_size, *values.shape[1:]))
                for keys, values in cache
            )
        if not hasattr(cache, "layers") or not all(
            hasattr(layer, "keys") and hasattr(layer, "values")
            for layer in cache.layers
        ):
            branch = copy.deepcopy(cache)
            branch.batch_repeat_interleave(batch_size)
            return branch
        branch = copy.copy(cache)
        branch.layers = []
        for original in cache.layers:
            layer = copy.copy(original)
            layer.keys = original.keys.expand(batch_size, *original.keys.shape[1:])
            layer.values = original.values.expand(batch_size, *original.values.shape[1:])
            branch.layers.append(layer)
        return branch

    @staticmethod
    def cache_length(cache):
        if hasattr(cache, "get_seq_length"):
            return cache.get_seq_length()
        if isinstance(cache, (tuple, list)) and cache:
            return cache[0][0].shape[-2]
        raise TypeError("unsupported prompt cache")

    def gradient_context(self):
        return contextlib.nullcontext() if self.train_backbone and self.training else torch.no_grad()

    def prepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise ValueError("native KV preparation expects one task")
        with self.gradient_context():
            output = self.backbone.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         use_cache=True)
        position = attention_mask.sum(dim=1) - 1
        task = output.last_hidden_state[0, position.item()]
        if not self.train_backbone or not self.training:
            task = task.detach().clone()
        self.task_prefills += 1
        return NativeKVContext(task, output.past_key_values,
                               int(attention_mask.sum().item()))

    def encode_suffixes(self, context: NativeKVContext, input_ids: torch.Tensor,
                        suffix_mask: torch.Tensor):
        if input_ids.shape != suffix_mask.shape or input_ids.ndim != 2:
            raise ValueError("suffix ids and mask must have shape [states, tokens]")
        count = input_ids.shape[0]
        cache = self.fork_prompt_cache(context.prompt_cache, count)
        prefix_mask = torch.ones(count, context.prefix_tokens,
                                 dtype=torch.bool, device=input_ids.device)
        with self.gradient_context():
            output = self.backbone.model(
                input_ids=input_ids,
                attention_mask=torch.cat((prefix_mask, suffix_mask), dim=1),
                past_key_values=cache,
                use_cache=True,
            )
        positions = suffix_mask.sum(dim=1) - 1
        vectors = output.last_hidden_state[
            torch.arange(count, device=input_ids.device), positions
        ]
        if not self.train_backbone or not self.training:
            vectors = vectors.detach().clone()
        if self.cache_length(context.prompt_cache) != context.prefix_tokens:
            raise RuntimeError("saved task KV was mutated by a state branch")
        self.suffix_forwards += 1
        return vectors

    def encode_standalone(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor):
        with self.gradient_context():
            output = self.backbone.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         use_cache=False)
        positions = attention_mask.sum(dim=1) - 1
        vectors = output.last_hidden_state[
            torch.arange(input_ids.shape[0], device=input_ids.device), positions
        ]
        return vectors if self.train_backbone and self.training else vectors.detach().clone()

    def frontier_step(self, context: NativeKVContext,
                      state_vectors: torch.Tensor, state_mask: torch.Tensor,
                      refutation_vectors: torch.Tensor,
                      refutation_mask: torch.Tensor,
                      action_vectors: torch.Tensor, action_mask: torch.Tensor,
                      coupling: str):
        if state_vectors.shape[0] != 1:
            raise ValueError("one task KV may update one frontier")
        if action_vectors.ndim == 2:
            actions = action_vectors[None, None].expand(
                1, state_vectors.shape[1], -1, -1)
        elif action_vectors.ndim == 4 and action_vectors.shape[:2] == state_vectors.shape[:2]:
            actions = action_vectors
        else:
            raise ValueError("actions must be shared [actions, hidden] or per-state [1, width, actions, hidden]")
        return self.cell(
            state_vectors.float(), state_mask,
            context.task_vector.float().unsqueeze(0),
            refutation_vectors.float(), refutation_mask,
            coupling=coupling, action_vectors=actions.float(),
            action_mask=action_mask,
            match_space=self.match_space,
        )
