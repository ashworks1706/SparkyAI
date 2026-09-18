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

`deploy/compose.prod.yml` overrides `compose.yml`: prebuilt images instead of builds, and no host ports except engine `:8080`. Phoenix, the datastores, Prometheus, and Grafana are reachable only over a tunnel. Put a reverse proxy with TLS in front of engine. `llama-server` runs on a GPU host; see `deploy/inference`.

## Models

Two `llama-server` containers, one per model: chat `:8000`, embeddings `:8001`. Locally `just model` starts them; in deployment run the same image on a GPU host.
Set `SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` accordingly. Details: `deploy/inference/README.md`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## The sandbox

`run_sandbox` needs a container runtime. Compose provides one as `sandboxd`, a `docker:27-dind`
daemon of its own with no host port: the engine reaches it over `DOCKER_HOST=tcp://sandboxd:2375`,
so a compromise of the engine reaches that daemon and not the host. The engine image carries only
the client. On a developer host running `just engine`, the local `docker` serves the same purpose
and nothing else is needed.

Commands run in `ghcr.io/ashworks1706/sparkyai-sandbox` (`deploy/docker/sandbox.Dockerfile`):
python3, jq and the usual text tools, and nothing that can fetch. Every container is started with
no network, a read-only root, dropped capabilities, a non-root user, and a `noexec` workspace, so
`python3 script.py` runs but `./script` does not.

The runtime is probed at boot. `sandbox.required = true` refuses to start without one; set it to
`false` to run without a sandbox, which leaves `run_sandbox` unregistered so the model is never
offered a tool that always fails.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust`, `sparkyai-scraper` and `sparkyai-sandbox` tagged `<sha>` and `main` on push to `main` — only the images whose inputs changed (`workflow_dispatch` rebuilds all). CI likewise runs only the units a change touches.

## Observability

- Traces, LLM generations, product events: self-hosted Phoenix, below. Every app exports OpenTelemetry to `SPARKY_TELEMETRY__PHOENIX_URL` (compose sets `SPARKY_PHOENIX_URL`); an empty URL disables export. Spans land in the project named by `telemetry.project_name`. A Discord conversation is one `session.id`, and each product event is a span of its own.
- Logs: pretty in development and JSON to stdout otherwise. The developer console also writes `.sparky/logs/`; deployed logs stay with the platform log driver.
- Database: `just db` starts pgweb on http://localhost:8081, loopback only. It browses the same database the engine reads and writes, so a change made there is a change to live data. `chunks.embedding` is a 1024-dimension vector and does not render usefully in a table.
- Metrics: `just metrics` starts Prometheus (:9090) and Grafana (:3000, dashboard **SparkyAI inference**), both on loopback only. They scrape `llama-server`, which exports Prometheus format on its own port; `chat` and `embed` run with `--metrics`. On a GPU host add `just gpu-metrics` for the utilisation, VRAM and temperature panels. The exporter shells out to `nvidia-smi`, so it runs under the nvidia container runtime with the `utility` driver capability rather than binding the driver library in by path.

Phoenix holds one `llm` span per model call: the full prompt, the full reply, token counts, and latency. It is the source the training pipeline reads. It holds the same spans as a trace tree too: one conversation, its model calls, tool calls and retrievals, in order and with timings. Prometheus holds server-side time series: throughput, queue depth, batching.

### Phoenix

```bash
just phoenix       # trace UI on http://localhost:6006, loopback
```

One container, `arizephoenix/phoenix:version-20.11.0`, data in the `phoenixdata` volume, UI and OTLP endpoint on the same port. Export is off until `SPARKY_TELEMETRY__PHOENIX_URL=http://localhost:6006` is in `.env`, so an app started without Phoenix never retries a dead endpoint. For the compose apps set `SPARKY_PHOENIX_URL=http://phoenix:6006`, which compose passes through. Spans carry OpenInference attributes beside the `gen_ai.*` ones because the Phoenix UI keys off those. It has no authentication by default and holds full prompts and replies, so in production it has no host port; reach it over a tunnel. Set `SPARKY_TELEMETRY__PHOENIX_API_KEY` when it does authenticate; every app and `just data export` send it as a bearer token.

`just data export` reads the `llm` spans of `telemetry.project_name` back through `GET /v1/projects/<project>/spans`, so it needs the same `SPARKY_TELEMETRY__PHOENIX_URL`.

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
