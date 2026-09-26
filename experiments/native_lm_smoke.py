"""Local no-download load/update/save/reload smoke for the frontier LM bridge."""

import json
import argparse
import tempfile
import time
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from interference_search.native_lm import NativeFrontierLM


def make_model(config):
    return NativeFrontierLM(Qwen3ForCausalLM(config), actions=7, heads=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    started = time.perf_counter()
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128,
    )
    model = make_model(config).to(device).train()
    task = torch.randint(1, 128, (2, 7))
    states = torch.randint(1, 128, (2, 3, 5))
    refs = torch.randint(1, 128, (2, 2, 5))
    state_mask = torch.tensor([[True, True, False], [True, True, True]])
    ref_mask = torch.tensor([[True, False], [True, True]])
    state_tokens = state_mask.unsqueeze(-1).expand_as(states)
    ref_tokens = ref_mask.unsqueeze(-1).expand_as(refs)
    kwargs = dict(
        task_ids=task, task_mask=torch.ones_like(task, dtype=torch.bool),
        state_ids=states, state_token_mask=state_tokens, state_mask=state_mask,
        refutation_ids=refs, refutation_token_mask=ref_tokens,
        refutation_mask=ref_mask,
        survival_targets=torch.tensor([[1, 0, 0], [0, 1, 1]]),
        action_targets=torch.tensor([[2, 5, -100], [1, 0, 4]]),
        successor_targets=torch.randn(2, 3, 7, 64),
        successor_mask=state_mask.unsqueeze(-1).expand(2, 3, 7),
        action_refutation_targets=torch.zeros(2, 3, 7),
        lm_labels=task,
    )
    kwargs = {name: value.to(device) for name, value in kwargs.items()}
    old_backbone = model.backbone.model.embed_tokens.weight.detach().clone()
    old_frontier = model.frontier_cell.query.weight.detach().clone()
    old_action_path = model.frontier_cell.action_query.weight.detach().clone()
    old_transition = model.frontier_cell.transition_head[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(**kwargs)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.backbone.model.embed_tokens.weight.grad is not None
    assert model.frontier_cell.query.weight.grad is not None
    assert model.frontier_cell.action_query.weight.grad is not None
    assert model.frontier_cell.transition_head[0].weight.grad is not None
    optimizer.step()
    assert not torch.equal(old_backbone, model.backbone.model.embed_tokens.weight.detach())
    assert not torch.equal(old_frontier, model.frontier_cell.query.weight.detach())
    assert not torch.equal(old_action_path, model.frontier_cell.action_query.weight.detach())
    assert not torch.equal(old_transition, model.frontier_cell.transition_head[0].weight.detach())

    model.eval()
    with torch.no_grad():
        expected = model(**kwargs).frontier.survival_logits
        task_vector, _ = model.encode_task(kwargs["task_ids"], kwargs["task_mask"])
        state_vectors = model.state_encoder(kwargs["state_ids"], kwargs["state_token_mask"])
        ref_vectors = model.state_encoder(kwargs["refutation_ids"], kwargs["refutation_token_mask"])
        cached = model.frontier_step(
            task_vector, state_vectors, kwargs["state_mask"], ref_vectors,
            kwargs["refutation_mask"],
        ).survival_logits
    torch.testing.assert_close(cached, expected)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native_frontier.pt"
        torch.save(model.state_dict(), path)
        reloaded = make_model(config).to(device).eval()
        reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        with torch.no_grad():
            actual = reloaded(**kwargs).frontier.survival_logits
    torch.testing.assert_close(actual, expected)
    print(json.dumps({
        "status": "passed",
        "device": device,
        "seed": 17,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": round(float(output.loss.detach()), 6),
        "backbone_changed": True,
        "frontier_changed": True,
        "action_path_changed": True,
        "transition_path_changed": True,
        "save_reload_equal": True,
        "cached_step_equal": True,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2) if device == "cuda" else None,
    }))


if __name__ == "__main__":
    main()
