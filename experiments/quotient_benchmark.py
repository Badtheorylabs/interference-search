"""Measure exact compact-state work and differentiable readout.

The hidden assignment is an oracle solely for certifying synthetic failures.
This benchmark is an architecture preflight, not an LLM result or speedup claim.
"""

import argparse
import json
import random
import statistics
import time

import torch

from interference_search.quotient_state import QuotientState, VerifiedRefutation


def timed(device, operation):
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    if device == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000


def run_case(decisions: int, device: str, seed: int):
    rng = random.Random(seed + decisions)
    hidden = tuple(rng.randrange(2) for _ in range(decisions))
    state = QuotientState(decisions)
    refutations = []
    for step in range(decisions // 2):
        start = (step * 2) % (decisions - 3)
        indices = (start, start + 1, start + 2)
        values = {index: rng.randrange(2) for index in indices}
        if all(values[index] == hidden[index] for index in indices):
            values[indices[0]] = 1 - hidden[indices[0]]
        refutations.append(VerifiedRefutation.from_mapping(values))

    def update():
        for item in refutations:
            state.refute(item, lambda proposed: any(
                hidden[index] != bit for index, bit in proposed.bindings
            ))

    _, update_ms = timed(device, update)
    logits = torch.randn(decisions, device=device, requires_grad=True)

    def readout():
        log_z = state.log_partition(logits)
        log_z.backward()
        return log_z

    log_z, cold_readout_ms = timed(device, readout)
    readout_samples = []
    for _ in range(5):
        logits.grad = None
        log_z, elapsed = timed(device, readout)
        readout_samples.append(elapsed)
    best, best_ms = timed(device, lambda: state.best(logits))
    assert state.contains(hidden) and best is not None and state.contains(best)
    assert torch.isfinite(log_z) and torch.isfinite(logits.grad).all()
    return {
        "decisions": decisions,
        "represented_assignments": state.count(),
        "reachable_nodes": state.reachable_nodes(),
        "certified_refutations": len(refutations),
        "update_ms": round(update_ms, 3),
        "cold_log_partition_backward_ms": round(cold_readout_ms, 3),
        "steady_log_partition_backward_ms": round(statistics.median(readout_samples), 3),
        "best_ms": round(best_ms, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    cases = [run_case(decisions, args.device, args.seed) for decisions in (12, 20, 28, 36)]
    print(json.dumps({
        "status": "architecture preflight only",
        "device": args.device,
        "seed": args.seed,
        "torch": torch.__version__,
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
