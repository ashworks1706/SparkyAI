# Sparky Models

Datasets, post-training, and evals. Python 3.12, managed with `uv`.

```bash
cd models
uv sync --extra dev                  # base + tooling
uv sync --extra train --extra eval   # on a GPU box
uv run sparky-data stats
uv run sparky-train sft --config configs/train/sft.yaml
uv run sparky-eval run tool_selection
```

| Layer | Choice |
|---|---|
| Training | TRL + PEFT; Unsloth on single GPU; accelerate, bitsandbytes |
| Data | HF `datasets`, JSONL trajectories; `openai` client for teacher generation |
| Evals | Inspect AI (our suites), BFCL (tool calling), lm-eval (regression) |
| Tracking | Weights & Biases |
| Releases | Hugging Face Hub |

Order: SFT → DPO → GRPO. See `docs/ROADMAP.md` Phase 6 and `docs/decisions/0003-posttraining-stack.md`.

BFCL is run from its own repo (`gorilla/berkeley-function-call-leaderboard`) against the vLLM endpoint; `sparky-eval bfcl` wraps that invocation.
