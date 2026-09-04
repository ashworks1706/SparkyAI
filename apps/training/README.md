# apps/training

Dataset preparation, deterministic engine evals, and GGUF model exports.

```bash
cd apps/training
uv sync --extra dev

just data export
just data verify
just data stats

just eval run
just eval baseline
just eval compare

just train sft --dry-run
just train sft
```

| Module | Holds |
|---|---|
| `datasets/export.py` | Phoenix `llm` spans → `TrainingExample` (full prompt + reply). Phoenix is the source because only the engine's `llm` span carries the whole prompt. |
| `datasets/redact.py` | Regex PII removal: emails, phones, Discord and ASU ids, bot tokens. |
| `datasets/verify.py` | Schema, non-empty replies, named tool calls, dedupe by content hash. |
| `evals/runner.py` | Posts each golden case to `/chat`, reads the engine's JSONL trace for that request. |
| `evals/suites/` | Deterministic scorers: tool selection and arguments, grounding (citation + mentions), refusal, permissions (policy decisions), clarification, memory (two-turn), latency. |
| `evals/cases/` | Hand-written ASU questions with expectations. Add a line, it runs. |
| `posttrain/sft.py` | Unsloth QLoRA + TRL, chat template from the base model, TensorBoard logs, GGUF export. |

Generated datasets, reports, checkpoints, TensorBoard logs, and exports are written under `../../.sparky/training/`. Engine traces are read from `../../.sparky/traces/`. Golden cases and the promoted baseline remain source-controlled in `evals/`.

Data export requires Phoenix. Evals require a live engine and use deterministic scorers rather than an LLM judge.
