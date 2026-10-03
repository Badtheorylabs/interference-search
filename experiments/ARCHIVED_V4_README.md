# Archived: V4 frontier-slot line

These experiments attached latent hypothesis slots to pretrained Qwen3-4B and
failed their objective. They are not part of the current roadmap and nothing
builds on them:

- `v4b_causal_probe.py`, `v4b_null_split_diagnostic.py`,
  `v4b_slot_retrieval_ko.py`, `build_v4b_decision_points.py` — V4b causal
  diagnostics; generation-path interventions measured KL ~1e-9, the
  retrieval-knockout result was never control-checked.
- The 68.5% decoding plateau against the 80% gate is the formal failure.

The successor direction is native latent interference, documented in
`docs/NATIVE_LATENT_INTERFERENCE_ROADMAP.md`, with the working lab in
`experiments/lab/`. The checkpoint
`runs/interference_qwen3_4b_v4a_families_800` stays frozen and untouched.
