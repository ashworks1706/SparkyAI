# apps/training

Datasets, post-training, and evals for Sparky models. Python 3.12, managed with `uv`. Runs on a GPU box occasionally; not a service.

```bash
cd apps/training
uv sync --extra dev                  # base + tooling
uv sync --extra train --extra eval   # on a GPU box
uv run data stats
uv run train sft --config configs/train/sft.yaml
uv run eval run tool_selection
```

| Layer | Choice |
|---|---|
| Post-training | TRL + PEFT; Unsloth on single GPU; accelerate, bitsandbytes |
| Data | HF `datasets`, JSONL trajectories; `openai` client for teacher generation |
| Evals | Inspect AI (our suites in `evals/suites`, cases in `evals/cases`), BFCL (tool calling), lm-eval (regression) |
| Tracking | Weights & Biases |
| Releases | Hugging Face Hub |

Order: SFT → DPO → GRPO. See `docs/ROADMAP.md` Phase 6 and `docs/decisions/0003-posttraining-stack.md`.

BFCL is run from its own repo (`gorilla/berkeley-function-call-leaderboard`) against the vLLM endpoint; `eval bfcl` wraps that invocation.
