# Deploy

## Local

```bash
just bootstrap     # .env, git hooks, deps, datastores (postgres, redis, minio)
just engine        # then in separate shells: just discord, just web
just up            # docker compose -f deploy/compose.yml up -d
```

Starts engine, discord, scraper, Postgres 17 (pgvector), Redis 7, MinIO. `engine` and `discord` are the same image (`rust.Dockerfile`) with different entrypoints. The models are behind the `model` profile: `just model` starts chat and embed. `just crawl` starts self-hosted Firecrawl (five containers, API on :3002) for the scraper;

## Production

```bash
git clone https://github.com/ashworks1706/SparkyAI.git && cd SparkyAI
cp .env.example .env               # secrets and per-machine URLs; SPARKY_APP__ENV=production
                                   # settings live in the committed sparky.toml
SPARKY_IMAGE_TAG=main just prod-up # pulls ghcr.io images; datastores have no host ports
just prod-logs engine
```

`deploy/compose.prod.yml` overrides `compose.yml`: prebuilt images instead of builds, and no host ports except engine `:8080`. PostHog, the datastores, Prometheus, and Grafana are reachable only over a tunnel. Put a reverse proxy with TLS in front of engine. `llama-server` runs on a GPU host; see `deploy/inference`.

## Models

Two `llama-server` containers, one per model: chat `:8000`, embeddings `:8001`. Locally `just model` starts them; in deployment run the same image on a GPU host.
Set `SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` accordingly. Details: `deploy/inference/README.md`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust` and `sparkyai-scraper` tagged `<sha>` and `main` on push to `main` — only the images whose inputs changed (`workflow_dispatch` rebuilds all). CI likewise runs only the units a change touches.

## Observability

- Traces, LLM analytics, product events: self-hosted PostHog, below. Every app exports OpenTelemetry to `SPARKY_TELEMETRY__HOST` (default `http://localhost:8010`; compose sets `http://posthog`) with `SPARKY_TELEMETRY__PROJECT_TOKEN`; an empty token disables export. Model spans become `$ai_generation` events; a Discord conversation is one `$ai_session_id`.
- Reading one conversation: Phoenix, below. The same spans also go to `SPARKY_TELEMETRY__PHOENIX_URL` when it is set, and the two destinations are independent.
- Logs: pretty in development and JSON to stdout otherwise. The developer console also writes `.sparky/logs/`; deployed logs stay with the platform log driver.
- Database: `just db` starts pgweb on http://localhost:8081, loopback only. It browses the same database the engine reads and writes, so a change made there is a change to live data. `chunks.embedding` is a 1024-dimension vector and does not render usefully in a table.
- Metrics: `just metrics` starts Prometheus (:9090) and Grafana (:3000, dashboard **SparkyAI inference**), both on loopback only. They scrape `llama-server`, which exports Prometheus format on its own port; `chat` and `embed` run with `--metrics`. On a GPU host add `just gpu-metrics` for the utilisation, VRAM and temperature panels. The exporter shells out to `nvidia-smi`, so it runs under the nvidia container runtime with the `utility` driver capability rather than binding the driver library in by path.

PostHog holds one `$ai_generation` per model call: the full prompt, the full reply, token counts, and latency. It is the source the training pipeline reads. Prometheus holds server-side time series: throughput, queue depth, batching. Phoenix holds the same spans as a trace tree: one conversation, its model calls, tool calls and retrievals, in order and with timings.

### Phoenix

```bash
just phoenix       # trace UI on http://localhost:6006, loopback
```

One container, `arizephoenix/phoenix:version-20.11.0`, data in the `phoenixdata` volume, UI and OTLP endpoint on the same port. Export is off until `SPARKY_TELEMETRY__PHOENIX_URL=http://localhost:6006` is in `.env`, so an app started without Phoenix never retries a dead endpoint. For the compose apps set `SPARKY_PHOENIX_URL=http://phoenix:6006`, which compose passes through. Spans carry OpenInference attributes beside the `gen_ai.*` ones because the Phoenix UI keys off those. It has no authentication and holds full prompts and replies, so in production it has no host port; reach it over a tunnel.

### PostHog

```bash
just posthog       # fetch pinned upstream files into .sparky/posthog, then start the posthog profile
```

The hobby stack of `github.com/PostHog/posthog` at the commit in `deploy/posthog/VERSION`, flattened into the `posthog-*` services of `compose.yml` (28 containers; session replay, error tracking, screenshots, and live events are left out). It wants about 16 GB of memory. `scripts/posthog.sh` sparse-checks-out that commit into `.sparky/posthog/src` (ClickHouse config, Kafka topics, Temporal and livestream config) and downloads GeoIP into `.sparky/posthog/share`; set `SPARKY_POSTHOG_DIR` to an absolute path to keep them elsewhere. Image pins live once, in the `x-posthog-images` block at the top of `compose.yml`.

The UI and every ingestion path sit behind `posthog` on http://localhost:8010, loopback only: `/i/v1/traces` (OTLP traces), `/batch/` (events); `/i/v0/ai/otel` takes PostHog's own payload rather than OTLP here, so `telemetry.ai_path` is empty. The first start runs migrations for several minutes; `curl -s localhost:8010/_health` returns 200 when it is up.

First run:

1. Open http://localhost:8010, create the account, organization, and project.
2. Put the project token (Settings, Project, Project token, `phc_...`) in `.env` as `SPARKY_TELEMETRY__PROJECT_TOKEN`.
3. For `just data export`, create a personal API key with the Query Read scope and set `SPARKY_TRAINING__POSTHOG_HOST=http://localhost:8010`, `SPARKY_TRAINING__POSTHOG_PROJECT_ID` (the number in the project URL), and `SPARKY_TRAINING__POSTHOG_API_KEY`.

`SPARKY_POSTHOG_SECRET` and `SPARKY_POSTHOG_ENCRYPTION_SALT_KEYS` have local-only defaults; set real values in `.env` before the first start on any shared host, and keep them afterwards (the salt encrypts stored data). `SPARKY_POSTHOG_SITE_URL` changes the URL PostHog puts in links. In production `posthog` has no host port.

### Reading the dashboard

| Panel | Metric | Reads as |
|---|---|---|
| Generation throughput | `rate(llamacpp:tokens_predicted_total[1m])` | tokens per wall-clock second, all requests combined |
| Prompt throughput | `llamacpp:prompt_tokens_total`, `..._cached_total` | prompt tokens evaluated vs. reused from cache |
| Queue | `llamacpp:requests_processing`, `..._deferred` | deferred above zero means requests are waiting for a slot |
| Batching efficiency | `llamacpp:n_busy_slots_per_decode` | near 1 with a non-empty queue means `--parallel` is too low |
| Prompt cache hit rate | cached / (cached + new) | a stable system prompt keeps this high |
| Server-reported speed | `llamacpp:predicted_tokens_seconds`, `..._prompt_tokens_seconds` | llama-server's own running average, independent of load |
| Context ceiling | `llamacpp:n_tokens_max` | the largest context one slot accepts |

Both servers start with two slots (`SPARKY_CHAT_PARALLEL`, `SPARKY_EMBED_PARALLEL`). Each slot takes a `--ctx-size / N` share of the context window, so raise both together.

Grafana's admin password comes from `SPARKY_GRAFANA_PASSWORD` (default `admin`). Both ports bind to `127.0.0.1`. Tunnel to reach them on a remote host; set a real password before exposing either.
