"""Countdown domain connecting a trained NativeKVFrontier to live search."""

from dataclasses import dataclass, field

import torch

from .async_execution import AsyncAction
from .countdown_actions import CountdownActionCodec


@dataclass
class NativeKVCountdownContext:
    problem: dict
    kv: object
    state_vectors: dict = field(default_factory=dict)
    refuted_states: list = field(default_factory=list)


class NativeKVCountdownDomain:
    def __init__(self, model, tokenizer, action_vectors, max_numbers=7,
                 actions_per_state=4, coupling="inhibit", retain_refutations=32):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.action_vectors = action_vectors
        self.codec = CountdownActionCodec(max_numbers)
        self.actions_per_state = actions_per_state
        self.coupling = coupling
        self.retain_refutations = retain_refutations
        self.device = next(model.parameters()).device
        if action_vectors.shape != (len(self.codec.actions), model.backbone.config.hidden_size):
            raise ValueError("action vectors do not match codec and model")

    @staticmethod
    def task_text(problem):
        return (f"Use each number {problem['numbers']} once with + - * / to reach "
                f"{problem['target']}. Intermediate values must be positive whole numbers.")

    @staticmethod
    def state_text(state):
        return "remaining values: " + ", ".join(map(str, state))

    def _tokens(self, texts, length):
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=length, add_special_tokens=False,
                               return_tensors="pt")
        return (batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device).bool())

    def start(self, problem):
        return tuple(sorted(problem["numbers"]))

    def key(self, state):
        return state

    @torch.inference_mode()
    def prepare(self, problem):
        ids, mask = self._tokens([self.task_text(problem)], 96)
        return NativeKVCountdownContext(problem, self.model.prepare(ids, mask))

    @torch.inference_mode()
    def _encode_missing(self, states, context):
        missing = list(dict.fromkeys(
            state for state in states if state not in context.state_vectors))
        if not missing:
            return
        ids, mask = self._tokens([self.state_text(state) for state in missing], 32)
        vectors = self.model.encode_suffixes(context.kv, ids, mask)
        for state, vector in zip(missing, vectors):
            context.state_vectors[state] = vector.detach()

    @torch.inference_mode()
    def propose_batch(self, states, context):
        refuted = context.refuted_states[-self.retain_refutations:]
        self._encode_missing(list(states) + list(refuted), context)
        state_vectors = torch.stack([context.state_vectors[state] for state in states])[None]
        hidden = state_vectors.shape[-1]
        refs = (torch.stack([context.state_vectors[state] for state in refuted])[None]
                if refuted else state_vectors.new_zeros((1, 0, hidden)))
        state_mask = torch.ones((1, len(states)), dtype=torch.bool, device=self.device)
        ref_mask = torch.ones((1, len(refuted)), dtype=torch.bool, device=self.device)
        legal = torch.tensor([self.codec.legal_mask(state) for state in states],
                             dtype=torch.bool, device=self.device)[None]
        output = self.model.frontier_step(
            context.kv, state_vectors, state_mask, refs, ref_mask,
            self.action_vectors, legal, self.coupling)
        combined = output.proposal_logits + output.survival_logits.unsqueeze(-1)
        batches = []
        for slot, state in enumerate(states):
            count = min(self.actions_per_state, int(legal[0, slot].sum()))
            chosen = torch.topk(combined[0, slot], count).indices.tolist()
            batches.append([
                AsyncAction(action, output.predicted_successors[0, slot, action].detach(),
                            float(combined[0, slot, action]))
                for action in chosen
            ])
        return batches

    def execute(self, state, action, context):
        child = self.codec.apply(state, action)
        if child is None:
            raise ValueError("model selected a masked action")
        return child

    def observe_execution(self, state, action, child, context):
        if self.certified_dead(child, context.problem) and child not in context.refuted_states:
            context.refuted_states.append(child)

    def observe_prediction(self, state, action, predicted, child, context):
        return None

    def snapshot_round(self, context):
        return {"refuted_states": tuple(context.refuted_states)}

    def observe_pruned(self, states, context):
        for state in states:
            if state not in context.refuted_states:
                context.refuted_states.append(state)
        if len(context.refuted_states) > self.retain_refutations:
            del context.refuted_states[:-self.retain_refutations]

    def is_goal(self, state, problem):
        return state == (problem["target"],)

    def certified_dead(self, state, problem):
        return len(state) == 1 and state[0] != problem["target"]
