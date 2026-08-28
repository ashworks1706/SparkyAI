# evals

Shared evaluation data. No code.

- `cases/` — JSONL per suite: input, expected tool / sources / behavior. Read by `models/` (Inspect runners) and by the backend's trace replay.

Suites: tool_selection, tool_args, grounding, memory, permissions, clarification, refusal, latency.
