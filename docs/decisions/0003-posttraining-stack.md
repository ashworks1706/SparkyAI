# 0003 — Post-training, dataset, and eval stack

**Training.** TRL (SFT/DPO/GRPO trainers) + PEFT (LoRA/QLoRA) on Qwen3-14B. Unsloth for single-GPU speed and memory. GRPO uses programmatic rewards only, with vLLM rollouts. No full fine-tuning, no learned reward model.

**Data.** JSONL trajectories in Qwen's native tool-call format. Sources: teacher generation against the Sparky tool schema, verified by schema + expected-tool checks; hand-written boundary cases; redacted production traces as they accumulate. `data/` is never committed.

**Evals.** Inspect AI for the eight Sparky suites (cases from `evals/cases`, scorers programmatic where possible). BFCL as the standing tool-calling benchmark on every model version. lm-eval for regression on a few general tasks. LLM-as-judge only for free-text quality, calibrated against a human-labeled subset.

**Tracking and release.** W&B for runs; merged weights + adapter + quantized variants to the Hugging Face Hub with dataset description and eval results.

**Compute.** RunPod A100 80GB for LoRA on 14B; H100 for GRPO.
