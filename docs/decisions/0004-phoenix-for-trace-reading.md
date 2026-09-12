# Phoenix runs beside PostHog for reading one conversation

2026-09-11.

PostHog answers what users do: Discord usage, people, funnels, retention, and the training export
([[0003-posthog]]). It does not answer what one conversation did. Its trace tree is the LLM
analytics view, and that view is fed by `$ai_generation` events, which a self-hosted stack never
ingests. Reading a conversation there means writing HogQL against `posthog.trace_spans`.

Phoenix renders a trace as a tree: the request, each model call with its prompt and reply, each
tool call, each retrieval, with timings. It holds nothing else, costs one container, and was the
backend before PostHog.

Both run. Every app exports every span to both, and each destination is independent: PostHog needs
`telemetry.host` and `telemetry.project_token`, Phoenix needs `telemetry.phoenix_url`, and export
runs while either is set. Phoenix takes no authentication, so its exporter sends no bearer header.
The path under `phoenix_url` is `/v1/traces`, fixed by OTLP, so it is a constant rather than a
setting; a Phoenix behind a prefix is reached by putting the prefix in `phoenix_url`.

## Span attributes

Phoenix ingests `gen_ai.*` spans but its UI keys off OpenInference attributes, so a span carrying
only `gen_ai.*` appears in the tree without its messages rendered. Spans therefore carry both sets.
The OpenInference keys duplicate values already in the `gen_ai.*` and `sparky.*` keys; both are
truncated by `agent.max_span_value_chars`.

| Span | OpenInference attributes |
|---|---|
| `agent.run`, `agent.resume` | `openinference.span.kind=CHAIN`, `input.value`, `output.value`, `session.id`, `user.id` |
| `llm`, `task` | `openinference.span.kind=LLM`, `input.value` and `output.value` as JSON with their mime types, `llm.model_name`, `llm.token_count.prompt`, `llm.token_count.completion`, `session.id`, `user.id` |
| `tool` | `openinference.span.kind=TOOL`, `tool.name`, `tool.call_id`, `input.value` (redacted arguments), `output.value` (redacted result), `session.id`, `user.id` |
| `retrieve` | `openinference.span.kind=RETRIEVER`, `input.value`, `output.value`, `session.id`, `user.id` |
| discord interaction spans | `openinference.span.kind=CHAIN`, `input.value`, `output.value`, `session.id` recorded with the conversation, `user.id` |

## Running it

`just phoenix` starts the `phoenix` profile: one container, the UI and the OTLP endpoint on
http://localhost:6006, loopback only, pinned to `arizephoenix/phoenix:version-20.11.0`, data in the
`phoenixdata` volume. Export is off until `SPARKY_TELEMETRY__PHOENIX_URL` is set, so an app started
without Phoenix running exports nothing there instead of retrying a dead endpoint. Compose passes
`SPARKY_PHOENIX_URL` through to the three apps. In production `phoenix` has no host port: it holds
full prompts and replies and has no authentication.

## What each is for

| Question | Where |
|---|---|
| what did this conversation do, step by step | Phoenix |
| how many students ask at exam time, and who returns | PostHog |
| training data from model calls | PostHog, `posthog.trace_spans` |
| is the model server keeping up | Grafana |
