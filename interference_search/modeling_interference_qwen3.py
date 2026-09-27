"""Qwen3 with a native persistent Interference Search state.

The hypothesis frontier is part of every configured decoder block and is
persisted on the standard KV cache object. Standard causal-LM logits therefore
depend on constructive cross-hypothesis exchange and signed verified-failure
updates without an external search controller.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3PreTrainedModel, Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from .configuration_interference_qwen3 import InterferenceQwen3Config


@dataclass
class InterferenceBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.Tensor = None
    past_key_values: Cache | None = None
    frontier_state: torch.Tensor | None = None
    constructive_norm: torch.Tensor | None = None
    destructive_norm: torch.Tensor | None = None
    slot_write_gate: torch.Tensor | None = None
    token_read_gate: torch.Tensor | None = None
    refutation_match: torch.Tensor | None = None
    slot_occupancy: torch.Tensor | None = None
    assignment_entropy: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class InterferenceCausalLMOutputWithPast(CausalLMOutputWithPast):
    frontier_state: torch.Tensor | None = None
    constructive_norm: torch.Tensor | None = None
    destructive_norm: torch.Tensor | None = None
    slot_write_gate: torch.Tensor | None = None
    token_read_gate: torch.Tensor | None = None
    refutation_match: torch.Tensor | None = None
    slot_occupancy: torch.Tensor | None = None
    assignment_entropy: torch.Tensor | None = None


class NativeHypothesisUpdate(nn.Module):
    """Causal hypothesis-register update inside one decoder block."""

    def __init__(self, config: InterferenceQwen3Config):
        super().__init__()
        hidden, dim = config.hidden_size, config.frontier_dim
        self.scale = config.interference_scale
        self.token_down = nn.Linear(hidden, dim, bias=False)
        self.token_up = nn.Linear(dim, hidden, bias=False)
        self.slot_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.constructive = nn.MultiheadAttention(
            dim, config.frontier_heads, dropout=0.0, batch_first=True)
        self.token_to_slots = nn.MultiheadAttention(
            dim, config.frontier_heads, dropout=0.0, batch_first=True)
        self.slot_identity = nn.Parameter(
            torch.randn(config.frontier_slots, dim) * config.initializer_range)
        self.slot_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.ref_query = nn.Linear(dim, dim, bias=False)
        self.ref_key = nn.Linear(dim, dim, bias=False)
        self.ref_value = nn.Linear(dim, dim, bias=False)
        self.readout = nn.MultiheadAttention(
            dim, config.frontier_heads, dropout=0.0, batch_first=True)
        self.write_gate = nn.Linear(dim * 2, dim)
        self.read_gate = nn.Linear(dim, 1)
        self.occupancy = nn.Linear(dim, 1)

    def forward(self, hidden_states, frontier_state, refutation_mask=None,
                coupling="inhibit"):
        native_dtype = self.token_down.weight.dtype
        compact_tokens = self.token_norm(self.token_down(
            hidden_states.to(native_dtype)))
        latest = compact_tokens[:, -1:]
        frontier_state = frontier_state.to(native_dtype)
        slots = self.slot_norm(frontier_state) + self.slot_identity.unsqueeze(0)
        positive, _ = self.constructive(slots, slots, slots, need_weights=False)
        token_write, assignment = self.token_to_slots(
            slots, compact_tokens, compact_tokens,
            need_weights=True, average_attn_weights=True)
        write = torch.sigmoid(self.write_gate(torch.cat((slots, token_write), dim=-1)))
        occupancy = torch.sigmoid(self.occupancy(slots))

        negative = torch.zeros_like(slots)
        matches = slots.new_zeros(slots.shape[0], slots.shape[1], hidden_states.shape[1])
        if refutation_mask is not None and refutation_mask.any():
            query = F.normalize(self.ref_query(slots).float(), dim=-1)
            keys = F.normalize(self.ref_key(compact_tokens).float(), dim=-1)
            logits = 4.0 * torch.matmul(query, keys.transpose(1, 2))
            matches = torch.sigmoid(logits).to(slots.dtype) * refutation_mask[:, None]
            values = self.ref_value(compact_tokens).float()
            negative = torch.matmul(matches, values)
            negative = negative / matches.sum(dim=-1, keepdim=True).clamp_min(1.0)
            negative = negative.to(slots.dtype)

        signed = positive
        if coupling == "inhibit":
            signed = signed - self.scale * negative
        elif coupling == "excite":
            signed = signed + self.scale * negative
        elif coupling != "zero":
            raise ValueError("coupling must be inhibit, zero, or excite")
        frontier_state = self.slot_norm(
            frontier_state + 0.1 * write * occupancy * (signed + token_write)
            + 0.01 * self.slot_identity.unsqueeze(0))
        frontier_state = self.slot_norm(
            frontier_state + 0.1 * self.slot_mlp(frontier_state))

        readable_slots = self.slot_norm(frontier_state) * occupancy
        token_update, _ = self.readout(latest, readable_slots,
                                       readable_slots, need_weights=False)
        read_gate = torch.sigmoid(self.read_gate(token_update))
        token_update = read_gate * token_update
        revised = hidden_states.clone()
        revised[:, -1:] = revised[:, -1:] + self.token_up(token_update).to(
            hidden_states.dtype)
        assignment_entropy = -(assignment.float().clamp_min(1e-12)
                               * assignment.float().clamp_min(1e-12).log()).sum(dim=-1)
        return (revised, frontier_state, positive, negative,
                write.mean(dim=-1), read_gate.squeeze(-1),
                matches.max(dim=-1).values, occupancy.squeeze(-1), assignment_entropy)


class InterferenceQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: InterferenceQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.interference_enabled = layer_idx % config.interference_layer_stride == 0
        self.hypothesis_update = (NativeHypothesisUpdate(config)
                                  if self.interference_enabled else None)

    def forward(self, hidden_states, frontier_state,
                refutation_mask=None, coupling="inhibit", **kwargs):
        hidden_states = super().forward(hidden_states, **kwargs)
        if self.hypothesis_update is None:
            zero = frontier_state.new_zeros(frontier_state.shape)
            slot_zero = frontier_state.new_zeros(frontier_state.shape[:2])
            read_zero = frontier_state.new_zeros(frontier_state.shape[0], 1)
            return (hidden_states, frontier_state, zero, zero, slot_zero,
                    read_zero, slot_zero, slot_zero, slot_zero)
        return self.hypothesis_update(
            hidden_states, frontier_state, refutation_mask, coupling)


class InterferenceQwen3Model(Qwen3PreTrainedModel):
    config_class = InterferenceQwen3Config
    _no_split_modules = ["InterferenceQwen3DecoderLayer"]

    def __init__(self, config: InterferenceQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            InterferenceQwen3DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.frontier_init = nn.Parameter(torch.randn(
            config.frontier_slots, config.frontier_dim) * config.initializer_range)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.post_init()
        self.reset_native_output()

    def reset_native_output(self):
        """Make a fresh architectural transplant exactly preserve base logits."""
        for layer in self.layers:
            if layer.hypothesis_update is not None:
                nn.init.zeros_(layer.hypothesis_update.token_up.weight)

    def set_native_dtype(self, dtype):
        """Keep the small recurrent state numerically stable beside a BF16 base."""
        self.frontier_init.data = self.frontier_init.data.to(dtype)
        for layer in self.layers:
            if layer.hypothesis_update is not None:
                layer.hypothesis_update.to(dtype=dtype)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                coupling="inhibit", refutation_mask=None,
                output_hidden_states=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch = inputs_embeds.shape[0]
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, past_seen + inputs_embeds.shape[1],
                dtype=torch.bool, device=inputs_embeds.device)
        raw_attention_mask = attention_mask
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config, "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            attention_mask = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                attention_mask["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs)

        if refutation_mask is None:
            if input_ids is not None and self.config.tool_error_token_id is not None:
                refutation_mask = input_ids == self.config.tool_error_token_id
            else:
                refutation_mask = torch.zeros(
                    batch, inputs_embeds.shape[1], dtype=torch.bool,
                    device=inputs_embeds.device)
        if refutation_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("refutation_mask must align with current input tokens")

        frontier_state = (getattr(past_key_values, "frontier_state", None)
                          if past_key_values is not None else None)
        if frontier_state is None:
            frontier_state = self.frontier_init.unsqueeze(0).expand(batch, -1, -1)
        hidden_states = inputs_embeds
        history = [hidden_states] if output_hidden_states else None
        positive_norms, negative_norms = [], []
        write_gates, read_gates, refutation_matches = [], [], []
        occupancies, assignment_entropies = [], []
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for index, layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            (hidden_states, frontier_state, positive, negative,
             write_gate, read_gate, refutation_match,
             occupancy, assignment_entropy) = layer(
                hidden_states, frontier_state,
                refutation_mask=refutation_mask, coupling=coupling,
                attention_mask=attention_mask[self.config.layer_types[index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, **kwargs)
            positive_norms.append(positive.float().norm(dim=-1).mean(dim=-1))
            negative_norms.append(negative.float().norm(dim=-1).mean(dim=-1))
            write_gates.append(write_gate.float())
            read_gates.append(read_gate.float())
            refutation_matches.append(refutation_match.float())
            occupancies.append(occupancy.float())
            assignment_entropies.append(assignment_entropy.float())
            if output_hidden_states:
                history.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            history.append(hidden_states)
        if use_cache:
            past_key_values.frontier_state = frontier_state
        return InterferenceBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            frontier_state=frontier_state,
            constructive_norm=torch.stack(positive_norms, dim=1),
            destructive_norm=torch.stack(negative_norms, dim=1),
            slot_write_gate=torch.stack(write_gates, dim=1),
            token_read_gate=torch.stack(read_gates, dim=1),
            refutation_match=torch.stack(refutation_matches, dim=1),
            slot_occupancy=torch.stack(occupancies, dim=1),
            assignment_entropy=torch.stack(assignment_entropies, dim=1),
            hidden_states=tuple(history) if history is not None else None)


class InterferenceQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = InterferenceQwen3Config
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: InterferenceQwen3Config):
        super().__init__(config)
        self.model = InterferenceQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.model.reset_native_output()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, coupling="inhibit",
                refutation_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            coupling=coupling, refutation_mask=refutation_mask,
            **kwargs)
        hidden = outputs.last_hidden_state
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, indices])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels,
                vocab_size=self.config.vocab_size, **kwargs)
        return InterferenceCausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            frontier_state=outputs.frontier_state,
            constructive_norm=outputs.constructive_norm,
            destructive_norm=outputs.destructive_norm,
            slot_write_gate=outputs.slot_write_gate,
            token_read_gate=outputs.token_read_gate,
            refutation_match=outputs.refutation_match,
            slot_occupancy=outputs.slot_occupancy,
            assignment_entropy=outputs.assignment_entropy)


__all__ = [
    "InterferenceQwen3Config", "InterferenceQwen3Model",
    "InterferenceQwen3ForCausalLM", "NativeHypothesisUpdate",
]
