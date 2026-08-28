# evals

Shared evaluation data. No code here; runners are in `src/training/evals`.

- `cases/` — JSONL per suite: input, expected tool / sources / behavior. Read by the Inspect runners and by the engine's trace replay.

Suites: tool_selection, tool_args, grounding, memory, permissions, clarification, refusal, latency.
