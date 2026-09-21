# apps/training

Dataset preparation, deterministic engine evals, and GGUF model exports.

```bash
just data export | verify | stats
just eval run | baseline | compare
just train sft --dry-run
just train sft
```

| Module | Holds |
|---|---|
| `datasets/export.py` | the engine's `llm` spans from Phoenix to `TrainingExample` (full prompt and reply) |
| `datasets/redact.py` | regex PII removal: emails, phones, Discord and ASU ids, bot tokens |
| `datasets/verify.py` | schema, non-empty replies, named tool calls, dedupe by content hash |
| `evals/runner.py` | posts each golden case to `/chat` and reads the engine's JSONL trace for that request |
| `evals/suites/` | deterministic scorers: tool selection and arguments, grounding, refusal, permissions, clarification, memory, latency |
| `evals/cases/` | hand-written ASU questions with expectations; add a line and it runs |
| `posttrain/sft.py` | Unsloth QLoRA and TRL, chat template from the base model, TensorBoard logs, GGUF export |

Outputs (datasets, reports, checkpoints, TensorBoard logs, exports) go under `.sparky/training/`. Engine traces are read from `.sparky/traces/`. Golden cases and the promoted baseline are committed in `evals/`.

Data export needs Phoenix and `SPARKY_TELEMETRY__PHOENIX_URL` in `.env`, plus `SPARKY_TELEMETRY__PHOENIX_API_KEY` when Phoenix authenticates; it reads the project named by `telemetry.project_name`. Evals need a live engine.
