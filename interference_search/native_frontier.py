"""Trainable one-pass frontier cell for the next Interference Search experiment.

This is a reference module, not a trained model. A backbone encodes the task and
each canonical state into vectors. Exact state merging and verifier-certified
hard exclusions happen outside this cell. The cell lets all live states attend
to one another, proposes actions for all of them, and applies a content-based
nonnegative penalty from verified failed regions to both state survival and
action proposal scores.

The same module can be trained with positive, zero, or negative coupling to
isolate the sign of the refutation pathway at matched parameter count.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FrontierOutput:
    state_vectors: torch.Tensor
    proposal_logits: torch.Tensor
    survival_logits: torch.Tensor
    refutation_match: torch.Tensor
    action_refutation_match: torch.Tensor
    action_refutation_logits: torch.Tensor
    predicted_successors: torch.Tensor


class NativeFrontierCell(nn.Module):
    """One synchronized update over a padded frontier of canonical states.

    ``state_vectors`` has shape [batch, width, hidden], ``task_vector`` is
    [batch, hidden], and ``refutations`` is [batch, failures, hidden]. Valid
    state and refutation masks are True. ``action_vectors`` may be supplied by
    a model proposer; otherwise learned action slots are used. Hard exclusion
    masks come from a verifier and are never inferred from similarity.
    """

    def __init__(self, hidden: int, actions: int, heads: int = 4, scale: float = 0.1):
        super().__init__()
        if hidden < 1 or actions < 1 or hidden % heads:
            raise ValueError("hidden must be divisible by heads; actions must be positive")
        if scale < 0:
            raise ValueError("scale must be nonnegative")
        self.hidden = hidden
        self.scale = scale
        self.task_projection = nn.Linear(hidden, hidden, bias=False)
        self.state_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.state_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.action_query = nn.Linear(hidden, hidden, bias=False)
        self.refutation_key = nn.Linear(hidden, hidden, bias=False)
        self.register_buffer("match_logit_bias", torch.tensor(1.0))
        self.value_head = nn.Linear(hidden, 1)
        self.action_slots = nn.Parameter(torch.randn(actions, hidden) * 0.02)
        self.transition_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.proposal_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        state_vectors: torch.Tensor,
        state_mask: torch.Tensor,
        task_vector: torch.Tensor,
        refutations: torch.Tensor,
        refutation_mask: torch.Tensor,
        hard_exclusion: torch.Tensor | None = None,
        coupling: str = "inhibit",
        action_vectors: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        hard_action_exclusion: torch.Tensor | None = None,
        match_space: str = "learned",
    ) -> FrontierOutput:
        if coupling not in {"inhibit", "zero", "excite"}:
            raise ValueError("coupling must be inhibit, zero, or excite")
        if match_space not in {"learned", "model"}:
            raise ValueError("match_space must be learned or model")
        batch, width, hidden = state_vectors.shape
        if hidden != self.hidden or task_vector.shape != (batch, hidden):
            raise ValueError("state and task vector shapes disagree with hidden size")
        if state_mask.shape != (batch, width) or not state_mask.any(dim=1).all():
            raise ValueError("each batch item needs at least one live state")
        if refutations.ndim != 3 or refutations.shape[0] != batch or refutations.shape[2] != hidden:
            raise ValueError("refutation vectors have wrong shape")
        if refutation_mask.shape != refutations.shape[:2]:
            raise ValueError("refutation mask has wrong shape")
        if hard_exclusion is not None and hard_exclusion.shape != (batch, width):
            raise ValueError("hard exclusion mask has wrong shape")
        if action_vectors is not None and (action_vectors.ndim != 4 or
                                           action_vectors.shape[:2] != (batch, width) or
                                           action_vectors.shape[-1] != hidden):
            raise ValueError("action vectors must have shape [batch, width, actions, hidden]")
        action_count = self.action_slots.shape[0] if action_vectors is None else action_vectors.shape[2]
        if action_count < 1:
            raise ValueError("at least one action slot is required")
        if action_mask is not None and action_mask.shape != (batch, width, action_count):
            raise ValueError("action mask has wrong shape")
        if hard_action_exclusion is not None and hard_action_exclusion.shape != (batch, width, action_count):
            raise ValueError("hard action exclusion has wrong shape")

        enriched = state_vectors + self.task_projection(task_vector).unsqueeze(1)
        shared, _ = self.state_attention(
            enriched, enriched, enriched,
            key_padding_mask=~state_mask,
            need_weights=False,
        )
        states = self.state_norm(enriched + shared)
        if action_vectors is None:
            actions = states.unsqueeze(2) + self.action_slots.view(1, 1, action_count, hidden)
        else:
            actions = action_vectors
        expanded_states = states.unsqueeze(2).expand(-1, -1, action_count, -1)
        action_context = torch.cat((expanded_states, actions), dim=-1)
        proposal = self.proposal_head(action_context).squeeze(-1).float()
        predicted_successors = self.transition_head(action_context)

        if refutations.shape[1]:
            if match_space == "model":
                keys = F.normalize(refutations.float(), dim=-1)
                state_queries = F.normalize(states.float(), dim=-1)
                action_queries = F.normalize(predicted_successors.float(), dim=-1)
            else:
                keys = F.normalize(self.refutation_key(refutations).float(), dim=-1)
                state_queries = F.normalize(self.query(states).float(), dim=-1)
                action_queries = F.normalize(self.action_query(predicted_successors).float(), dim=-1)
            state_logits = 4.0 * torch.matmul(state_queries, keys.transpose(1, 2)) - self.match_logit_bias.float()
            action_logits = 4.0 * torch.einsum("bwah,brh->bwar", action_queries, keys) - self.match_logit_bias.float()
            matches = torch.sigmoid(state_logits) * refutation_mask[:, None]
            action_matches = torch.sigmoid(action_logits) * refutation_mask[:, None, None]
            # Sum rather than softmax: two independent refutations can suppress
            # more than one. The mask keeps absent entries from contributing.
            penalty = matches.sum(dim=-1)
            action_penalty = action_matches.sum(dim=-1)
        else:
            penalty = torch.zeros(batch, width, device=states.device, dtype=torch.float32)
            action_penalty = torch.zeros(batch, width, action_count, device=states.device, dtype=torch.float32)
            action_logits = torch.zeros(batch, width, action_count, 0, device=states.device, dtype=torch.float32)

        survival = self.value_head(states).squeeze(-1).float()
        if coupling == "inhibit":
            survival = survival - self.scale * penalty
            proposal = proposal - self.scale * action_penalty
        elif coupling == "excite":
            survival = survival + self.scale * penalty
            proposal = proposal + self.scale * action_penalty
        invalid = ~state_mask if hard_exclusion is None else (~state_mask | hard_exclusion)
        survival = survival.masked_fill(invalid, -1e9)
        invalid_actions = invalid.unsqueeze(-1)
        if action_mask is not None:
            invalid_actions = invalid_actions | ~action_mask
        if hard_action_exclusion is not None:
            invalid_actions = invalid_actions | hard_action_exclusion
        proposal = proposal.masked_fill(invalid_actions, -1e9)
        return FrontierOutput(states, proposal, survival, penalty, action_penalty, action_logits,
                              predicted_successors)
