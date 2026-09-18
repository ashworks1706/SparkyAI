# evals

Source-controlled evaluation inputs.

- `cases/` contains JSONL cases and expected behavior. Every `*.jsonl` under it is loaded.
  - `asu_golden.jsonl` is the original set: grounding, refusal, permissions, memory, latency.
  - `tool_routing.jsonl` is the regression set for tool choice and answer voice: which source
    answers a question, and that the answer never names a tool or a source key.
- `baseline.json` stores accepted suite rates when one has been promoted.

Generated reports go to `.sparky/training/evals/` at the repository root.

## Suites

`tool_selection` and `tool_args` read the trace: which tool ran and what it was given.
`grounding` reads the answer and its citations, matching a citation by its source key rather
than its label. `voice` fails an answer that names its own machinery, which is what a student
can neither act on nor verify. `memory`, `permissions`, `clarification`, `refusal` and
`latency` are unchanged.

Run one with `just eval run --suite voice`.
