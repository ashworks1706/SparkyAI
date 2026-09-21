# Deploy

## Local

```bash
just bootstrap     # .env, git hooks, deps, datastores (postgres, redis, minio), schema
just engine        # then in separate shells: just discord, just web
just up            # the compose stack, built locally
```

`just up` starts engine, discord, scraper, Postgres 17 (pgvector), Redis 7, and MinIO. `engine` and `discord` share one image (`rust.Dockerfile`) with different entrypoints. `just model` starts the chat and embed models (profile `model`). `just crawl` starts self-hosted Firecrawl for the scraper (five containers, API on :3002). `just search` starts SearXNG on :8888.

## Production

```bash
git clone https://github.com/ashworks1706/SparkyAI.git && cd SparkyAI
cp .env.example .env               # secrets and per-machine URLs; SPARKY_APP__ENV=production
SPARKY_IMAGE_TAG=main just prod-up # pulls ghcr.io images; datastores have no host ports
just prod-logs engine
```

`deploy/compose.prod.yml` overrides `compose.yml` with prebuilt images and no host ports except engine `:8080`. Phoenix, the datastores, Prometheus, and Grafana are reachable only over a tunnel. Put a reverse proxy with TLS in front of the engine. Settings live in the committed `sparky.toml`.

## Models

Two `llama-server` containers: chat `:8000` and embeddings `:8001`. `just model` uses the CUDA image and reserves an NVIDIA device, so it needs a GPU and the container toolkit. Without one:

- `just model-cpu` applies `deploy/compose.cpu.yml`: the plain image, no device reservation, the same GGUFs on the processor. Set `SPARKY_CHAT_NGL=0` and `SPARKY_EMBED_NGL=0`.
- Or point `SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` at any OpenAI-compatible endpoint, with their API keys.

Details: `deploy/inference/README.md`.

## Web

`apps/web` builds to static files (`npm run build` to `apps/web/dist`). Deploy to Vercel (root directory `apps/web`) or any static host. It is not part of a Docker image.

## The sandbox

`run_sandbox` needs a container runtime. Compose provides `sandboxd`, a `docker:27-dind` daemon with no host port; the engine reaches it over `DOCKER_HOST=tcp://sandboxd:2375`, and the engine image carries only the client. On a developer host running `just engine`, the local `docker` serves.

Commands run in `ghcr.io/ashworks1706/sparkyai-sandbox` (`deploy/docker/sandbox.Dockerfile`) with a read-only root, dropped capabilities, a non-root user, and a `noexec` workspace, so `python3 script.py` runs but `./script` does not. With `sandbox.egress` off a container has no network; with it on, it reaches public HTTP and HTTPS only through `sparkyai-sandbox-proxy` (`deploy/sandbox/squid.conf`). `just sandbox-images` builds both images locally; `just sandbox-down` removes the containers the engine started.

The runtime is probed at boot. `sandbox.required = true` refuses to start without one; `false` leaves `run_sandbox` unregistered.

## Images

On push to `main`, CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust`, `sparkyai-scraper`, `sparkyai-sandbox` and `sparkyai-sandbox-proxy`, tagged `<sha>` and `main`, only for images whose inputs changed (`workflow_dispatch` rebuilds all).

## Observability

- Traces, LLM generations, product events: self-hosted Phoenix, below. Every app exports OpenTelemetry to `SPARKY_TELEMETRY__PHOENIX_URL` (compose sets it from `SPARKY_PHOENIX_URL`); an empty URL disables export. Spans land in the project named by `telemetry.project_name`. A Discord conversation is one `session.id`, and each product event is its own span.
- Logs: pretty in development, JSON to stdout otherwise. The developer console also writes `.sparky/logs/`; deployed logs stay with the platform log driver.
- Database: `just db` starts pgweb on http://localhost:8081, loopback only. It edits live data. `chunks.embedding` does not render usefully in a table.
- Metrics: `just metrics` starts Prometheus (:9090) and Grafana (:3000, dashboard **SparkyAI inference**), both on loopback. They scrape `llama-server` (`chat` and `embed` run with `--metrics`). On a GPU host `just gpu-metrics` adds utilisation, VRAM and temperature panels; the exporter runs under the nvidia container runtime with the `utility` driver capability.

Phoenix holds one `llm` span per model call (full prompt, full reply, token counts, latency) inside a trace tree per conversation; the training pipeline reads these spans. Prometheus holds server-side time series: throughput, queue depth, batching.

### Phoenix

```bash
just phoenix       # trace UI on http://localhost:6006, loopback
```

One container, `arizephoenix/phoenix:version-20.11.0`, data in the `phoenixdata` volume, UI and OTLP endpoint on one port. Export is off until `SPARKY_TELEMETRY__PHOENIX_URL=http://localhost:6006` is in `.env`. For the compose apps set `SPARKY_PHOENIX_URL=http://phoenix:6006`. Phoenix has no authentication by default and holds full prompts and replies, so in production it has no host port. When it does authenticate, set `SPARKY_TELEMETRY__PHOENIX_API_KEY`; every app and `just data export` send it as a bearer token. `just data export` reads `llm` spans through `GET /v1/projects/<project>/spans`.

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

Grafana's admin password comes from `SPARKY_GRAFANA_PASSWORD` (compose default `admin`). Both ports bind to `127.0.0.1`; set a real password before exposing either.
