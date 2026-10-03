# Real SWE frontier gate

Objective: test whether the native Qwen3-4B Interference Search model can rank successful repository-work branches above failed siblings using real tool actions and observations.

Unit: one synchronized frontier snapshot from one SWE-Smith task. Two to eight trajectories for the same repository task form the live branches. Snapshots are taken at 25%, 50%, and 75% of each trajectory's executable steps.

Source: `SWE-bench/SWE-smith-trajectories`, revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, MIT license, from the locally hashed BTL-4 download. Only tasks with at least one resolved and one unresolved trajectory are admitted.

Labels: final `resolved` outcome is an exact trajectory-level verifier label. Tool observations are real recorded environment results. A failed final trajectory refutes that complete candidate path; it does not prove that every nearby state is dead.

Splits: deterministic repository-family hashing. A repository family appears in only one split. This prevents trajectories from the same codebase crossing train and evaluation.

Quality bar:

- At least one resolved and one unresolved branch.
- Every branch has a non-empty state, action, and following observation.
- No exact duplicate transition inside one frontier.
- Maximum eight branches, balanced between resolved and unresolved where possible.
- No inferred merge labels. Hashes record exact normalized equality only.
- Raw source remains untouched; projections retain source revision and trajectory IDs.

Deliverables: train, validation, and test JSONL; three golden rows; schema; deterministic build report.
