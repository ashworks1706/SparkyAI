# PostHog replaces Phoenix for traces, LLM observability, and product analytics

2026-09-11.

Phoenix held the OpenTelemetry spans and was the source of training data. It shows traces and
nothing else: no product analytics for the Discord bot or the web app, no users, no funnels.
PostHog, self-hosted, takes the same OTLP spans, turns model spans into LLM generations tied to a
user and a session, and holds product events in the same place. Prometheus and Grafana stay for
llama-server metrics; the self-hosted PostHog stack has no metrics ingestion.

## Running it

PostHog runs from the hobby stack of the upstream repository, pinned to one commit, vendored into
`deploy/compose.yml` under the `posthog` profile. `just posthog` fetches the pinned upstream files
the stack mounts (ClickHouse config, Kafka topics, GeoIP) into `.sparky/posthog` and starts it.
The Caddy proxy is the `posthog` service; the rest are named `posthog-<upstream name>`. They
reach each other over a `posthog` network by their upstream names. The upstream hobby file omits `capture-ai`, which serves `/i/v0/ai/*`; the
vendored stack includes it. The UI and every ingestion path are behind one Caddy proxy on
`http://localhost:8010`, loopback only. The stack wants about 16 GB of memory.

After the first start, create a project in the UI and put its project token in `.env`.

## Settings

| Setting | Where | Meaning |
|---|---|---|
| `telemetry.host` | `.env`, default `http://localhost:8010` | PostHog base URL; compose sets `http://posthog` |
| `telemetry.project_token` | `.env`, secret | project token; empty disables export and logs one warning |
| `telemetry.traces_path` | `sparky.toml` | `/i/v1/traces` |
| `telemetry.ai_path` | `sparky.toml` | `/i/v0/ai/otel` |
| `telemetry.sample_ratio`, `export_timeout_secs`, `service_name`, `span_target_prefix` | `sparky.toml` | unchanged |
| `analytics.*` | `sparky.toml` | Discord product events, below |
| `cli.posthog_url` | `sparky.toml` | UI the console probes and opens |
| `training.posthog_host`, `posthog_project_id`, `posthog_api_key` | `.env`, key is a secret | HogQL query API for data export; the key is a personal API key with Query Read |
| `training.posthog_page_rows` | `sparky.toml` | rows per HogQL page |

`telemetry.otlp_endpoint` and `cli.phoenix_url` and `training.phoenix_url` are removed.

## Export

Each process installs one tracer provider with two batch OTLP/HTTP protobuf exporters, both
authenticated with `Authorization: Bearer <project_token>`: `host + traces_path` and
`host + ai_path`. Every exported span goes to both. The traces endpoint keeps every span as a
distributed trace. The AI endpoint keeps spans with `gen_ai.*` attributes: a span with
`gen_ai.operation.name = chat` becomes an `$ai_generation` event, other spans become `$ai_span`,
and the root becomes `$ai_trace`. W3C `traceparent` still joins discord, engine, and scraper.

## Span attributes

OpenInference attributes (`openinference.span.kind`, `input.value`, `output.value`, `llm.*`,
`session.id`, `user.id`) are replaced.

| Span | Attributes |
|---|---|
| every span that knows them | `$ai_session_id` = conversation id, `posthog.distinct_id` = the user id the engine uses |
| `llm` (engine loop model call) | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.request.max_tokens`, `gen_ai.request.temperature`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.input.messages` (the request messages as the engine Message JSON array), `gen_ai.output.messages` (a one element array of the reply Message JSON), `sparky.span=llm`, `sparky.tools` (tool definitions offered), `sparky.step`, `sparky.attempt` |
| `task` (compaction, profile, summary calls) | as `llm`, with `sparky.span=task` and `sparky.task` |
| `agent.run`, `agent.resume` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name=sparky`, `sparky.input`, `sparky.output`, `sparky.status`, `sparky.request_id`, `sparky.tenant_id` |
| `tool` | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` (redacted), `gen_ai.tool.call.result` (redacted, truncated) |
| `retrieve` | `gen_ai.operation.name=retrieval`, `sparky.input`, `sparky.output` |
| discord interaction spans | `$ai_session_id`, `posthog.distinct_id`, `sparky.input`, `sparky.output`, `discord.command` |
| scraper source runs and live queries | `sparky.*` only; traces endpoint |

`gen_ai.provider.name` is `telemetry.provider_name` in `sparky.toml`. Span values are truncated to
`agent.max_span_value_chars` as before. Secrets and credentials never enter an attribute.

## Discord analytics

The bot sends product events to `host + /batch/` with the project token, from a bounded queue
flushed in the background. A full queue or a failed flush drops the batch and logs it; a reply
never waits on analytics. `distinct_id` is the Discord user id, the same one the engine sees.

| Event | Properties |
|---|---|
| `discord_ask` | `guild_id`, `channel_id`, `place` (channel, thread, dm), `private` |
| `discord_mention` | `guild_id`, `channel_id`, `place` |
| `discord_answer` | `$session_id`/`conversation_id`, `status`, `latency_ms`, `chars`, `place` |
| `discord_error` | `stage`, `kind` |
| `discord_reset`, `discord_memory`, `discord_forget`, `discord_confirm` | `guild_id`, `channel_id`, command result counts |

Settings under `[analytics]`: `enabled`, `queue_capacity`, `max_batch`, `flush_ms`.

## Training export

`apps/training` reads `$ai_generation` events where `properties.sparky.span = 'llm'` through
`POST /api/projects/<id>/query/` with HogQL, paged by timestamp (OFFSET is refused for personal
API keys), `posthog_page_rows` at a time. `$ai_input` holds the prompt messages and
`$ai_output_choices` the reply. The event uuid is the example id; `$ai_session_id` and the
distinct id carry through as before.
