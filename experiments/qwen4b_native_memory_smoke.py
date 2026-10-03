"""One-step H100 memory preflight for Qwen3-4B-Base plus native frontier.

Uses random inputs and discards the updated model. This verifies engineering
feasibility only, not trained behavior. No benchmark data enters this script.
"""

import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from interference_search.native_lm import NativeFrontierLM


DEFAULT_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--optimizer", choices=("adamw8bit", "none"), default="adamw8bit")
    parser.add_argument("--task-tokens", type=int, default=64)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--state-tokens", type=int, default=16)
    parser.add_argument("--refutations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if min(args.task_tokens, args.states, args.state_tokens, args.refutations, args.steps) < 1:
        parser.error("all shape arguments must be positive")
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = NativeFrontierLM(base, actions=168).to(device="cuda", dtype=torch.bfloat16).train()
    torch.cuda.reset_peak_memory_stats()
    loaded_seconds = time.perf_counter() - started
    vocab = base.config.vocab_size
    hidden = base.config.hidden_size
    task = torch.randint(1, vocab, (1, args.task_tokens), device="cuda")
    states = torch.randint(1, vocab, (1, args.states, args.state_tokens), device="cuda")
    refs = torch.randint(1, vocab, (1, args.refutations, args.state_tokens), device="cuda")
    kwargs = dict(
        task_ids=task,
        task_mask=torch.ones_like(task, dtype=torch.bool),
        state_ids=states,
        state_token_mask=torch.ones_like(states, dtype=torch.bool),
        state_mask=torch.ones(1, args.states, device="cuda", dtype=torch.bool),
        refutation_ids=refs,
        refutation_token_mask=torch.ones_like(refs, dtype=torch.bool),
        refutation_mask=torch.ones(1, args.refutations, device="cuda", dtype=torch.bool),
        survival_targets=torch.randint(0, 2, (1, args.states), device="cuda"),
        action_targets=torch.randint(0, 168, (1, args.states), device="cuda"),
        successor_targets=torch.randn(1, args.states, 168, hidden, device="cuda", dtype=torch.bfloat16),
        action_refutation_targets=torch.zeros(1, args.states, 168, device="cuda"),
    )
    backbone_parameter = model.backbone.model.layers[0].self_attn.q_proj.weight
    transition_parameter = model.frontier_cell.transition_head[0].weight
    before_backbone = backbone_parameter.detach().clone()
    before_transition = transition_parameter.detach().clone()
    optimizer = None
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
    step_times = []
    backward_times = []
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        output = model(**kwargs)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("non-finite or missing loss")
        output.loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - step_started)
        backbone_gradient = float(backbone_parameter.grad.float().abs().sum())
        transition_gradient = float(transition_parameter.grad.float().abs().sum())
        if not backbone_gradient > 0 or not transition_gradient > 0:
            raise RuntimeError("backbone or transition gradient is zero")
        if optimizer is not None:
            optimizer.step()
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_started)
    print(json.dumps({
        "status": "one-step memory preflight; random inputs; no behavior claim",
        "model": args.model,
        "revision": args.revision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "optimizer": args.optimizer,
        "task_tokens": args.task_tokens,
        "states": args.states,
        "state_tokens": args.state_tokens,
        "refutations": args.refutations,
        "steps": args.steps,
        "load_seconds": round(loaded_seconds, 2),
        "first_step_seconds": round(step_times[0], 2),
        "steady_step_seconds": round(statistics.median(step_times[1:]), 2) if len(step_times) > 1 else None,
        "steady_backward_seconds": round(statistics.median(backward_times[1:]), 2) if len(backward_times) > 1 else None,
        "total_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "loss": round(float(output.loss.detach()), 6),
        "backbone_gradient_sum": round(backbone_gradient, 6),
        "transition_gradient_sum": round(transition_gradient, 6),
        "backbone_changed": not torch.equal(before_backbone, backbone_parameter.detach()),
        "transition_changed": not torch.equal(before_transition, transition_parameter.detach()),
    }, indent=2))


if __name__ == "__main__":
    main()
