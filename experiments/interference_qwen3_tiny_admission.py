"""Fast admission for the self-contained Interference Qwen3 model class."""

import json
import tempfile
from pathlib import Path

import torch

from interference_search.configuration_interference_qwen3 import InterferenceQwen3Config
from interference_search.modeling_interference_qwen3 import InterferenceQwen3ForCausalLM


def main():
    torch.manual_seed(7)
    config = InterferenceQwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128,
        frontier_slots=4, frontier_dim=16, frontier_heads=4,
        tool_error_token_id=99, eos_token_id=2, pad_token_id=0,
        layer_types=["full_attention", "full_attention"])
    model = InterferenceQwen3ForCausalLM(config)
    ids = torch.tensor([[1, 4, 5, 6, 99], [1, 8, 9, 10, 11]])
    mask = torch.ones_like(ids)
    zero = model(ids, attention_mask=mask, use_cache=True, coupling="zero")
    inhibit = model(ids, attention_mask=mask, use_cache=True, coupling="inhibit")
    initial_delta = float((zero.logits - inhibit.logits).abs().max().detach())
    assert initial_delta == 0
    assert inhibit.frontier_state.shape == (2, 4, 16)
    assert inhibit.past_key_values.get_seq_length() == 5
    prior_frontier = inhibit.frontier_state.detach().clone()
    next_ids = torch.tensor([[12], [13]])
    continued = model(next_ids, attention_mask=torch.ones(2, 6, dtype=torch.long),
                      past_key_values=inhibit.past_key_values, use_cache=True)
    assert continued.past_key_values.get_seq_length() == 6
    assert not torch.allclose(prior_frontier, continued.frontier_state)

    trainable = [parameter for name, parameter in model.named_parameters()
                 if "hypothesis_update" in name or "frontier_init" in name]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    trained = model(ids, attention_mask=mask,
                    use_cache=False, coupling="inhibit")
    next_targets = torch.tensor([17, 18])
    loss = torch.nn.functional.cross_entropy(
        trained.logits[:, -1].float(), next_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, parameter.detach())
               for old, parameter in zip(before, trainable))
    with torch.no_grad():
        learned_zero = model(ids, attention_mask=mask, use_cache=False,
                             coupling="zero")
        learned_inhibit = model(ids, attention_mask=mask, use_cache=False,
                                coupling="inhibit")
    causal_delta = float((learned_zero.logits - learned_inhibit.logits)
                         .abs().max().detach())
    assert causal_delta > 0

    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = InterferenceQwen3ForCausalLM.from_pretrained(directory).eval()
        with torch.no_grad():
            expected = model(ids, attention_mask=mask, use_cache=False).logits
            actual = loaded(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(expected, actual)
        generated = loaded.generate(ids[:1, :3], max_new_tokens=3,
                                    do_sample=False, use_cache=True)
        assert generated.shape == (1, 6)

    report = {
        "status": "tiny self-contained Interference Qwen3 admitted",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "native_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_base_preserving_delta": initial_delta,
        "causal_max_logit_delta": causal_delta,
        "cache_frontier_shape": list(continued.frontier_state.shape),
        "cache_sequence_length": continued.past_key_values.get_seq_length(),
        "backward_update": True, "save_reload_equal": True,
        "standard_generate": True,
    }
    Path("results/interference_qwen3_tiny_admission.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
