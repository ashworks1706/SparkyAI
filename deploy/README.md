# Deploy

## Local

```bash
just bootstrap     # .env, git hooks, deps, datastores (postgres, redis, minio)
just engine        # then in separate shells: just discord, just web
just up            # docker compose -f deploy/compose.yml up -d
```

Starts engine, discord, scraper, Phoenix (trace UI at http://localhost:6006), Postgres 17 (pgvector), Redis 7, MinIO. `engine` and `discord` are the same image (`rust.Dockerfile`) with different entrypoints. The models are behind the `model` profile: `just model` starts chat and embed. `just crawl` starts self-hosted Firecrawl (five containers, API on :3002) for the scraper; `just browser` starts Playwright MCP (:8931, loopback) for the engine's browser tools; set `SPARKY_MCP__PLAYWRIGHT_URL=http://localhost:8931/mcp` for a host-run engine (the compose engine has it already).

## Production

```bash
git clone https://github.com/ashworks1706/SparkyAI.git && cd SparkyAI
cp .env.example .env               # production values; SPARKY_APP__ENV=production
SPARKY_IMAGE_TAG=main just prod-up # pulls ghcr.io images; datastores have no host ports
just prod-logs engine
```

`deploy/compose.prod.yml` overrides `compose.yml`: prebuilt images instead of builds, no host ports except engine `:8080`. Put a reverse proxy with TLS in front of engine. `llama-server` runs on a GPU host; see `deploy/inference`.

## Models

Two `llama-server` containers, one per model: chat `:8000`, embeddings `:8001`. Locally `just model` starts them; in deployment run the same image on a GPU host.
Set `SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` accordingly. Details: `deploy/inference/README.md`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust` and `sparkyai-scraper` tagged `<sha>` and `main` on push to `main` — only the images whose inputs changed (`workflow_dispatch` rebuilds all). CI likewise runs only the units a change touches.

## Observability

- Traces: every app exports OpenTelemetry to Phoenix (`SPARKY_TELEMETRY__OTLP_ENDPOINT`, default `http://localhost:4317`; empty disables). UI at http://localhost:6006. Engine spans carry OpenInference attributes; a Discord conversation is one Phoenix session.
- Logs: pretty in development and JSON to stdout otherwise. The developer console also writes `.sparky/logs/`; deployed logs stay with the platform log driver.
- Metrics: `just metrics` starts Prometheus (:9090) and Grafana (:3000, dashboard **SparkyAI inference**), both on loopback only. They scrape `llama-server`, which exports Prometheus format on its own port because `chat` and `embed` run with `--metrics`. On a GPU host add `just gpu-metrics` for utilisation and VRAM.

Phoenix and Grafana answer different questions and neither replaces the other. Phoenix holds one span per model call with the full prompt, the full reply, token counts, and latency — per request, and the source the training pipeline reads. Prometheus holds server-side time series — throughput, queue depth, batching — which is what tells you whether a slow request was slow to decode or was waiting for a slot.

### Reading the dashboard

| Panel | Metric | What it means |
|---|---|---|
| Generation throughput | `rate(llamacpp:tokens_predicted_total[1m])` | tokens per wall-clock second, all requests combined |
| Prompt throughput | `llamacpp:prompt_tokens_total`, `..._cached_total` | prompt tokens evaluated vs. reused from cache |
| Queue | `llamacpp:requests_processing`, `..._deferred` | deferred above zero means requests are waiting for a slot |
| Batching efficiency | `llamacpp:n_busy_slots_per_decode` | near 1 with a non-empty queue means `--parallel` is too low |
| Prompt cache hit rate | cached / (cached + new) | a stable system prompt should keep this high |

`llama-server` starts with one slot. Requests serialize until `--parallel N` is set in `compose.yml`, and each slot takes a `--ctx-size / N` share of the context window, so raise both together.

Grafana's admin password comes from `SPARKY_GRAFANA_PASSWORD` (default `admin`). Both ports bind to `127.0.0.1`; tunnel to reach them on a remote host, and set a real password before exposing either.
