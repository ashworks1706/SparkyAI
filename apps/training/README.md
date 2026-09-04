# apps/training

Datasets from real interactions, engine evals with a baseline gate, and SFT that exports a
GGUF llama-server can load. `core/` holds settings, types, and tests.

```bash
cd apps/training && uv sync --extra dev          # add --extra train for a GPU box

just data export                                 # Phoenix `llm` spans → data/raw (redacted)
just data verify                                 # → data/processed/sft.jsonl (schema, dedupe)
just data stats

just eval run                                    # golden cases → live engine → evals/last.json
just eval baseline                               # promote last.json to evals/baseline.json
just eval compare                                # fail on any suite regression

just train sft --dry-run                         # validate config + data, no GPU
just train sft                                   # QLoRA on Qwen3-4B → outputs/sft/gguf/*.gguf
SPARKY_CHAT_GGUF=outputs/sft/gguf/<file>.gguf just model   # serve it, then `just eval run` again
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

The runner needs the engine running from the repo root so `traces/` is shared, and Phoenix
up for export. No suite uses an LLM judge; every score is reproducible from the trace.
`train sft` has not yet been executed on this machine — `--dry-run` has.
