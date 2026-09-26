"""Reference language-model bridge for a native Interference Search frontier.

The backbone parses the task once. A small shared encoder reads each canonical
state and verified refutation; NativeFrontierCell proposes actions and scores
all live states in one recurrent round. This is a correctness reference for a
future 4B run, not an optimized serving implementation.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .native_frontier import FrontierOutput, NativeFrontierCell


@dataclass
class NativeLMOutput:
    loss: torch.Tensor | None
    frontier: FrontierOutput


class StateTextEncoder(nn.Module):
    def __init__(self, embeddings: nn.Module, hidden: int, heads: int = 4, max_length: int = 1024):
        super().__init__()
        self.embeddings = embeddings
        self.position = nn.Embedding(max_length, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.shape != token_mask.shape or token_ids.ndim != 3:
            raise ValueError("state tokens and mask must have shape [batch, slots, length]")
        batch, slots, length = token_ids.shape
        if not 1 <= length <= self.position.num_embeddings:
            raise ValueError("state text length is outside the encoder limit")
        if slots == 0:
            return self.embeddings.weight.new_zeros(batch, 0, self.embeddings.embedding_dim)
        flat_ids = token_ids.reshape(batch * slots, length)
        flat_mask = token_mask.reshape(batch * slots, length)
        safe_mask = flat_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.embeddings(flat_ids) + self.position(positions).unsqueeze(0)
        hidden = self.layers(hidden, src_key_padding_mask=~safe_mask)
        hidden = self.norm(hidden)
        pooled = (hidden * flat_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / flat_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return pooled.reshape(batch, slots, -1)


class NativeFrontierLM(nn.Module):
    """Attach a trainable synchronized frontier to a decoder-only HF backbone."""

    def __init__(self, backbone: nn.Module, actions: int, heads: int = 4):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.state_encoder = StateTextEncoder(backbone.get_input_embeddings(), hidden, heads)
        self.frontier_cell = NativeFrontierCell(hidden, actions, heads)

    def encode_task(self, task_ids: torch.Tensor, task_mask: torch.Tensor):
        """Encode a task once; reuse its vector across search rounds at inference."""
        if task_ids.shape != task_mask.shape or not task_mask.any(dim=1).all():
            raise ValueError("each task needs at least one token")
        body = self.backbone.model(input_ids=task_ids, attention_mask=task_mask)
        positions = torch.arange(task_ids.shape[1], device=task_ids.device)
        positions = positions.unsqueeze(0).expand_as(task_mask).masked_fill(~task_mask, -1).max(dim=1).values
        vector = body.last_hidden_state[
            torch.arange(task_ids.shape[0], device=task_ids.device), positions
        ]
        return vector, body.last_hidden_state

    def frontier_step(
        self,
        task_vector: torch.Tensor,
        state_vectors: torch.Tensor,
        state_mask: torch.Tensor,
        refutation_vectors: torch.Tensor,
        refutation_mask: torch.Tensor,
        hard_exclusion: torch.Tensor | None = None,
        coupling: str = "inhibit",
        action_vectors: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        hard_action_exclusion: torch.Tensor | None = None,
    ) -> FrontierOutput:
        """Run one level after encoding only newly created states/refutations."""
        return self.frontier_cell(
            state_vectors, state_mask, task_vector, refutation_vectors,
            refutation_mask, hard_exclusion=hard_exclusion, coupling=coupling,
            action_vectors=action_vectors, action_mask=action_mask,
            hard_action_exclusion=hard_action_exclusion,
        )

    def forward(
        self,
        task_ids: torch.Tensor,
        task_mask: torch.Tensor,
        state_ids: torch.Tensor,
        state_token_mask: torch.Tensor,
        state_mask: torch.Tensor,
        refutation_ids: torch.Tensor,
        refutation_token_mask: torch.Tensor,
        refutation_mask: torch.Tensor,
        hard_exclusion: torch.Tensor | None = None,
        survival_targets: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        lm_labels: torch.Tensor | None = None,
        coupling: str = "inhibit",
        action_vectors: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        hard_action_exclusion: torch.Tensor | None = None,
        successor_targets: torch.Tensor | None = None,
        successor_mask: torch.Tensor | None = None,
        action_refutation_targets: torch.Tensor | None = None,
    ) -> NativeLMOutput:
        task_vector, task_hidden = self.encode_task(task_ids, task_mask)
        states = self.state_encoder(state_ids, state_token_mask)
        refutations = self.state_encoder(refutation_ids, refutation_token_mask)
        frontier = self.frontier_step(
            task_vector, states, state_mask, refutations, refutation_mask,
            hard_exclusion=hard_exclusion, coupling=coupling,
            action_vectors=action_vectors, action_mask=action_mask,
            hard_action_exclusion=hard_action_exclusion,
        )

        losses = []
        valid = state_mask if hard_exclusion is None else (state_mask & ~hard_exclusion)
        valid_actions = valid.unsqueeze(-1).expand_as(frontier.proposal_logits)
        if action_mask is not None:
            valid_actions = valid_actions & action_mask
        if hard_action_exclusion is not None:
            valid_actions = valid_actions & ~hard_action_exclusion
        if survival_targets is not None:
            if survival_targets.shape != state_mask.shape:
                raise ValueError("survival targets have wrong shape")
            if valid.any():
                losses.append(F.binary_cross_entropy_with_logits(
                    frontier.survival_logits[valid].float(), survival_targets[valid].float()
                ))
        if action_targets is not None:
            if action_targets.shape != state_mask.shape:
                raise ValueError("action targets have wrong shape")
            trainable = valid & (action_targets >= 0)
            if trainable.any():
                chosen = action_targets[trainable]
                if chosen.max() >= frontier.proposal_logits.shape[-1]:
                    raise ValueError("action target exceeds available action slots")
                if action_mask is not None and not action_mask[trainable].gather(1, chosen[:, None]).all():
                    raise ValueError("action target points to a masked action")
                if hard_action_exclusion is not None and hard_action_exclusion[trainable].gather(1, chosen[:, None]).any():
                    raise ValueError("action target points to a hard excluded action")
                losses.append(F.cross_entropy(
                    frontier.proposal_logits[trainable].float(), chosen
                ))
        if successor_targets is not None:
            if successor_targets.shape != frontier.predicted_successors.shape:
                raise ValueError("successor targets have wrong shape")
            trainable_successors = valid_actions if successor_mask is None else valid_actions & successor_mask
            if trainable_successors.any():
                losses.append(F.smooth_l1_loss(
                    frontier.predicted_successors[trainable_successors].float(),
                    successor_targets.detach()[trainable_successors].float(),
                ))
        if action_refutation_targets is not None:
            if action_refutation_targets.shape != frontier.action_refutation_match.shape:
                raise ValueError("action refutation targets have wrong shape")
            if valid_actions.any():
                losses.append(F.mse_loss(
                    frontier.action_refutation_match[valid_actions].float(),
                    action_refutation_targets.float()[valid_actions],
                ))
        if lm_labels is not None:
            if lm_labels.shape != task_ids.shape:
                raise ValueError("language-model labels have wrong shape")
            logits = self.backbone.lm_head(task_hidden[:, :-1])
            losses.append(F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                lm_labels[:, 1:].reshape(-1), ignore_index=-100,
            ))
        return NativeLMOutput(sum(losses) if losses else None, frontier)
