# apps/training

Deterministic engine evals with a baseline gate.

```bash
just eval run | baseline | compare
```

| Module | Holds |
|---|---|
| `evals/runner.py` | posts each golden case to `/chat` and reads the engine's JSONL trace for that request |
| `evals/suites/` | deterministic scorers: tool selection and arguments, grounding, refusal, permissions, clarification, memory, latency |
| `evals/cases/` | hand-written ASU questions with expectations; add a line and it runs |

Reports go under `.sparky/training/evals/`. Engine traces are read from `.sparky/traces/`. Golden cases and the promoted baseline are committed in `evals/`. Evals need a live engine.

## Fine-tuning

Dataset export and post-training live in [loupe](https://github.com/ashworks1706/loupe). It reads SparkyAI's `llm` spans from Phoenix and returns a GGUF:

```bash
loupe data export --name sparky-sft --source phoenix --url http://127.0.0.1:6006 --project sparky --redact-extra asu-id
loupe data verify --name sparky-sft
loupe train sft experiments/sparky-sft/sft.yaml
```

Serve the result with `SPARKY_CHAT_GGUF=<file> just model`, then gate it here with `just eval run` and `just eval compare`.
