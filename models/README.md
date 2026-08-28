# Sparky Models

Python post-training track. Starts in Phase 6, after the eval suites exist.

- `datasets/` — generation and cleaning of tool-use, retrieval, clarification, and boundary examples
- `training/` — SFT (LoRA) → DPO → GRPO on verifiable rewards
- `evals/` — model-level evaluation against the fixed suites
