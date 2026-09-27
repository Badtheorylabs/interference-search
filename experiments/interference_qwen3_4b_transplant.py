"""Transplant Qwen3-4B weights into the native Interference Qwen3 class."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM


def tensor_sha(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config_values = base_config.to_dict()
    config_values.pop("model_type", None)
    config_values.pop("architectures", None)
    config = InterferenceQwen3Config(
        **config_values, frontier_slots=8, frontier_dim=128,
        frontier_heads=4, interference_scale=1.0,
        interference_layer_stride=1)
    config.architectures = ["InterferenceQwen3ForCausalLM"]

    model, loading = InterferenceQwen3ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", output_loading_info=True)
    model.eval()
    native_named = [(name, parameter) for name, parameter in model.named_parameters()
                    if "hypothesis_update" in name or "frontier_init" in name]
    native_parameters = [parameter for _, parameter in native_named]
    native_names = {name for name, _ in native_named}
    missing_native = set(loading["missing_keys"])
    non_native_missing = sorted(missing_native - native_names)

    stock = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    prompts = [
        "Write a Python function that returns the larger of two integers.",
        "A test failed because the empty list was not handled. Repair the function.",
    ]
    batch = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        stock_logits = stock(**batch, use_cache=False).logits
        native_logits = model(**batch, use_cache=False, coupling="zero").logits
    parity_delta = float((stock_logits - native_logits).abs().max())
    del stock, stock_logits, native_logits
    torch.cuda.empty_cache()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in native_parameters:
        parameter.requires_grad_(True)
    before = tensor_sha(native_parameters)
    optimizer = torch.optim.AdamW(native_parameters, lr=2e-4)
    refutation_mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    refutation_mask[:, -1] = True
    targets = tokenizer([" pass", " fix"], add_special_tokens=False,
                        return_tensors="pt", padding=True)["input_ids"][:, 0].cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    output = model(**batch, use_cache=False, coupling="inhibit",
                   refutation_mask=refutation_mask)
    loss = F.cross_entropy(output.logits[:, -1].float(), targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(native_parameters, 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    after = tensor_sha(native_parameters)

    model.eval()
    with torch.no_grad():
        zero = model(**batch, use_cache=False, coupling="zero",
                     refutation_mask=refutation_mask)
        inhibit = model(**batch, use_cache=True, coupling="inhibit",
                        refutation_mask=refutation_mask)
        causal_delta = float((zero.logits - inhibit.logits).abs().max())
        next_ids = inhibit.logits[:, -1].argmax(dim=-1, keepdim=True)
        continued = model(
            input_ids=next_ids,
            attention_mask=torch.ones(
                next_ids.shape[0], batch["input_ids"].shape[1] + 1,
                dtype=torch.long, device="cuda"),
            past_key_values=inhibit.past_key_values, use_cache=True)
        generated = model.generate(
            batch["input_ids"][:1], attention_mask=batch["attention_mask"][:1],
            max_new_tokens=2, do_sample=False, use_cache=True)

    args.out.mkdir(parents=True, exist_ok=True)
    native_path = args.out / "native_interference_state.pt"
    torch.save({name: parameter.detach().cpu()
                for name, parameter in native_named}, native_path)
    config.save_pretrained(args.out)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    native_count = sum(parameter.numel() for parameter in native_parameters)
    report = {
        "status": "Qwen3-4B weights transplanted into self-contained Interference Qwen3",
        "source_commit": args.source_commit, "base_model": args.model,
        "total_parameters": total_parameters,
        "native_parameters": native_count,
        "base_compatible_parameters": total_parameters - native_count,
        "native_fraction": native_count / total_parameters,
        "frontier_slots": config.frontier_slots,
        "frontier_dim": config.frontier_dim,
        "interference_layers": config.num_hidden_layers,
        "missing_keys": len(loading["missing_keys"]),
        "non_native_missing_keys": non_native_missing,
        "unexpected_keys": loading["unexpected_keys"],
        "pretrain_logit_max_delta": parity_delta,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "native_weights_changed": before != after,
        "post_update_causal_logit_delta": causal_delta,
        "cached_frontier_shape": list(continued.frontier_state.shape),
        "cached_sequence_length": continued.past_key_values.get_seq_length(),
        "standard_generate_shape": list(generated.shape),
        "step_seconds": step_seconds,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "native_state_path": str(native_path),
        "native_state_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
