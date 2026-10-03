"""Model-native Countdown adapter for the asynchronous search runtime.

The backbone encodes the task once. New canonical states are encoded once and
cached. The frontier head scores legal action slots for a whole ready batch.
The exact action codec executes selected actions, then certified terminal
failures can be supplied as refutation vectors to the next model call.

This is an inference bridge, not evidence that the head has learned to rank
actions or predict useful successors.
"""

import asyncio
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from .async_execution import AsyncAction
from .countdown_actions import CountdownActionCodec


@dataclass
class NativeCountdownContext:
    problem: dict
    task_vector: torch.Tensor
    prefix_token_ids: list[int] = field(default_factory=list)
    prompt_cache: Any = None
    state_vectors: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict)
    refuted_states: list[tuple[int, ...]] = field(default_factory=list)
    prediction_events: list[tuple[tuple[int, ...], int, torch.Tensor,
                                  tuple[int, ...]]] = field(default_factory=list)


class NativeAsyncCountdownDomain:
    def __init__(self, model, tokenizer, max_numbers: int = 7, actions_per_state: int = 4,
                 coupling: str = "zero", state_mode: str = "auto",
                 prefix_cache_min_work: int = 768, external_executor=None):
        if actions_per_state < 1:
            raise ValueError("actions_per_state must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.codec = CountdownActionCodec(max_numbers)
        if self.model.frontier_cell.action_slots.shape[0] != len(self.codec.actions):
            raise ValueError("model action slots must match the Countdown codec")
        self.actions_per_state = actions_per_state
        self.coupling = coupling
        if state_mode not in {"auto", "backbone_prefix", "backbone_full", "lightweight"}:
            raise ValueError("unknown native state encoder mode")
        if prefix_cache_min_work < 1:
            raise ValueError("prefix cache threshold must be positive")
        self.state_mode = state_mode
        self.prefix_cache_min_work = prefix_cache_min_work
        self.external_executor = external_executor
        self.device = next(model.parameters()).device
        self.task_backbone_forwards = 0
        self.state_backbone_forwards = 0
        self.lightweight_state_calls = 0
        self.encoder_routes = {"backbone_full": 0, "backbone_prefix": 0}

    def start(self, problem):
        return tuple(sorted(problem["numbers"]))

    def key(self, state):
        return state

    def _ids(self, texts):
        encoded = [self._tokenize(text) for text in texts]
        length = max(map(len, encoded))
        ids = torch.zeros((1, len(texts), length), dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for slot, row in enumerate(encoded):
            ids[0, slot, :len(row)] = torch.tensor(row, device=self.device)
            mask[0, slot, :len(row)] = True
        return ids, mask

    def _tokenize(self, text):
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(text)
        return ids or [1]

    def _prepare_sync(self, problem):
        text = problem.get("task_text") or (
            f"Numbers: {sorted(problem['numbers'])}. Target: {problem['target']}.")
        ids, mask = self._ids([text])
        with torch.inference_mode():
            if self.state_mode == "lightweight":
                vector, _ = self.model.encode_task(ids[:, 0], mask[:, 0])
                cache = None
            else:
                body = self.model.backbone.model(input_ids=ids[:, 0],
                                                 attention_mask=mask[:, 0],
                                                 use_cache=True)
                vector = body.last_hidden_state[:, -1]
                cache = body.past_key_values if self.state_mode in {
                    "auto", "backbone_prefix"} else None
            self.task_backbone_forwards += 1
        return NativeCountdownContext(problem, vector.detach(),
                                      ids[0, 0].tolist(), cache)

    async def prepare(self, problem):
        return await asyncio.to_thread(self._prepare_sync, problem)

    @staticmethod
    def _fork_prompt_cache(cache, batch_size):
        """Share immutable prefix KV tensors, copying only mutable layer wrappers.

        Transformers 4.57 DynamicCache layers replace ``keys`` and ``values``
        during update. Shallow layer copies therefore keep the saved prompt
        untouched while ``expand`` avoids an eager GPU copy of each prefix.
        Unknown cache implementations fall back to a conservative deep copy.
        """
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

    @torch.inference_mode()
    def _encode_new_states(self, states, context):
        missing = list(dict.fromkeys(state for state in states
                                     if state not in context.state_vectors))
        if missing:
            selected_mode = self._select_encoder_mode(len(context.prefix_token_ids),
                                                       len(missing))
            if selected_mode == "lightweight":
                ids, mask = self._ids([str(state) for state in missing])
                vectors = self.model.state_encoder(ids, mask)[0]
                self.lightweight_state_calls += 1
            else:
                token_rows = [self._tokenize(str(state)) for state in missing]
                if selected_mode == "backbone_full":
                    token_rows = [context.prefix_token_ids + row for row in token_rows]
                length = max(map(len, token_rows))
                ids = torch.zeros((len(missing), length), dtype=torch.long,
                                  device=self.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for index, row in enumerate(token_rows):
                    ids[index, :len(row)] = torch.tensor(row, device=self.device)
                    mask[index, :len(row)] = True
                kwargs = {"input_ids": ids, "use_cache": selected_mode == "backbone_prefix"}
                if selected_mode == "backbone_prefix":
                    if context.prompt_cache is None:
                        raise ValueError("prefix mode requires a prepared prompt cache")
                    cache = self._fork_prompt_cache(context.prompt_cache, len(missing))
                    prefix_mask = torch.ones((len(missing), len(context.prefix_token_ids)),
                                             dtype=torch.bool, device=self.device)
                    kwargs["attention_mask"] = torch.cat((prefix_mask, mask), dim=1)
                    kwargs["past_key_values"] = cache
                else:
                    kwargs["attention_mask"] = mask
                output = self.model.backbone.model(**kwargs)
                vectors = output.last_hidden_state[
                    torch.arange(len(missing), device=self.device),
                    mask.sum(dim=1) - 1,
                ]
                self.state_backbone_forwards += 1
                self.encoder_routes[selected_mode] += 1
            for state, vector in zip(missing, vectors):
                context.state_vectors[state] = vector.detach()

    def _select_encoder_mode(self, prefix_tokens: int, states: int) -> str:
        if self.state_mode != "auto":
            return self.state_mode
        work = prefix_tokens * states
        return "backbone_prefix" if work >= self.prefix_cache_min_work else "backbone_full"

    def _propose_sync(self, states, context):
        refuted = tuple(context.refuted_states)
        with torch.inference_mode():
            self._encode_new_states(states + list(refuted), context)
            state_vectors = torch.stack([context.state_vectors[state]
                                         for state in states], dim=0)[None]
            hidden = state_vectors.shape[-1]
            if refuted:
                refutation_vectors = torch.stack([context.state_vectors[state]
                                                   for state in refuted], dim=0)[None]
            else:
                refutation_vectors = state_vectors.new_zeros((1, 0, hidden))
            state_mask = torch.ones((1, len(states)), dtype=torch.bool, device=self.device)
            refutation_mask = torch.ones((1, len(refuted)), dtype=torch.bool,
                                          device=self.device)
            action_mask = torch.tensor([self.codec.legal_mask(state) for state in states],
                                       dtype=torch.bool, device=self.device)[None]
            output = self.model.frontier_step(
                context.task_vector, state_vectors, state_mask,
                refutation_vectors, refutation_mask,
                action_mask=action_mask, coupling=self.coupling,
            )
            result = []
            for slot, state in enumerate(states):
                legal = action_mask[0, slot]
                count = min(self.actions_per_state, int(legal.sum().item()))
                if not count:
                    result.append([])
                    continue
                chosen = torch.topk(output.proposal_logits[0, slot], count).indices.tolist()
                result.append([AsyncAction(
                    index, output.predicted_successors[0, slot, index].detach().clone(),
                    float(output.proposal_logits[0, slot, index])
                ) for index in chosen])
            return result

    async def propose_batch(self, states, context):
        return await asyncio.to_thread(self._propose_sync, states, context)

    async def execute(self, state, action, context):
        if self.external_executor is None:
            child = self.codec.apply(state, action)
            if child is None:
                raise ValueError("native proposer selected an illegal action")
            return child
        value = self.external_executor(state, action, context)
        return await value if inspect.isawaitable(value) else value

    def execution_key(self, state, action, context):
        return (state, action) if self.external_executor is None else None

    def is_goal(self, state, problem):
        return state == (problem["target"],)

    def certified_dead(self, state, problem):
        return len(state) == 1 and state[0] != problem["target"]

    def observe_execution(self, state, action, child, context):
        if self.certified_dead(child, context.problem) and child not in context.refuted_states:
            context.refuted_states.append(child)

    def observe_prediction(self, state, action, predicted, child, context):
        context.prediction_events.append((state, action, predicted, child))
