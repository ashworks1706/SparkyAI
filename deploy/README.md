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

## RunPod

`deploy/docker/runpod.Dockerfile` builds `ghcr.io/ashworks1706/sparkyai-runpod`: all of Sparky in one GPU pod. A RunPod pod cannot start containers, so supervisord (`deploy/runpod/supervisord.conf`) runs chat and embed `llama-server`, the engine, the bot, `scraper serve`, Postgres 16 with pgvector (the Ubuntu 24.04 package), Redis, MinIO and SearXNG as processes, and runs `scraper migrate` at each boot. It differs from compose in three ways:

- `run_sandbox` is off (`SPARKY_SANDBOX__ENABLED=false`).
- The scraper uses the `http` fetcher; there is no Firecrawl.
- Phoenix is not included; export stays off until `SPARKY_TELEMETRY__PHOENIX_URL` points at one.

Create the pod with the image, a volume at `/workspace` (database, object store, model cache, admin session, daily `pg_dump` in `backups/`), a host CUDA version of at least 12.8, and these environment variables: `SPARKY_DISCORD__TOKEN`, `SPARKY_DISCORD__GUILD_ID`, `SPARKY_ENGINE__SERVICE_TOKEN`, `SEARXNG_SECRET`, `SPARKY_APP__ENV=production`, and `HF_TOKEN` (a read token; community hosts share an IP that the Hugging Face Hub rate-limits without one). Every service listens on `127.0.0.1`; the pod exposes no port. The first boot downloads the GGUFs into `/workspace/models`.

### Current deployment

As of 2026-10-08.

| | |
|---|---|
| Pod | `sparky`, id `gdrbqqc1vsw6cv`, community cloud, always on |
| GPU | 1x NVIDIA GeForce RTX 3070, 7840 MiB VRAM, driver 580.65.06, host CUDA 13.0 |
| Host | 22 vCPU and 24 GB RAM allocated by RunPod |
| Disk | 30 GB container disk (wiped on restart), 40 GB persistent volume at `/workspace` |
| Image | `ghcr.io/ashworks1706/sparkyai-runpod:590f3a1ac7d18a26738c0ecbe3988e9fa113aeb5` |
| Ports | none exposed; every service binds `127.0.0.1` |

GPU memory, from `nvidia-smi` on the pod:

| Process | Model | Context | VRAM |
|---|---|---|---|
| chat | `Qwen/Qwen3-4B-GGUF:Q4_K_M` | 16384, 2 slots of 8192 | 3892 MiB |
| embed | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | 4096, 2 slots of 2048 | 1846 MiB |
| total | | | 5753 of 7840 MiB |

The RTX 3070 was the cheapest community GPU in stock with host CUDA 12.8 or newer that holds both models. The next options were the RTX 3080 and RTX A4000 at $0.17/hr and the RTX 3090 at $0.22/hr. A larger chat model or context needs one of those.

Cost:

| | |
|---|---|
| Compute | $0.13/hr: $3.12/day, about $95/month |
| Storage | 70 GB at RunPod's $0.10/GB-month rate, about $7/month |
| Credits | about $450 at deploy time, about 4.4 months at this rate |
| Spend to date | the RunPod console Billing page, or the `list-billing` call of the RunPod MCP server |

Metrics: both `llama-server` processes run with `--metrics` on `127.0.0.1`, but the pod runs no Prometheus, Grafana or Phoenix, so nothing collects them. Read GPU memory with `nvidia-smi` (see below); the console GPU reading showed 0 while 5.7 GB was in use.

### Operating the pod

The live deploy is one always-on community RTX 3070 pod named `sparky` at $0.13/hr (about $95 a month). It runs whether or not anyone talks to the bot, because the bot holds a Discord gateway connection. Billing stops only when the pod is stopped or terminated.

| Task | How |
|---|---|
| Deploy a change | Merge to `main`; CD publishes `sparkyai-runpod:<sha>` when its inputs change. Set the pod image to that tag. Changing the image restarts the pod. |
| Change a secret | Edit the pod environment in the RunPod console. The pod restarts with the new values. |
| Read logs | Pod logs in the console. Every service writes to the container log; supervisord prefixes its own state changes. |
| Restart | Restart the pod. `start.sh` reruns, supervisord starts every service, and `scraper migrate` applies new migrations. |
| Check the GPU | Override the start command with `nvidia-smi; exec /usr/local/bin/sparky-start`. The console GPU memory reading is not reliable. |

State lives on the host disk under `/workspace`: Postgres, Redis, MinIO, the GGUF cache, and the daily `pg_dump` in `backups/` (the newest `SPARKY_BACKUP_KEEP`, default 7). Community pods cannot attach network volumes, so if the host fails, the pod and its backups are lost together. To recover, create a new pod as above. The first boot initializes an empty database and downloads the models again; the scraper's scheduled sources rebuild the index.

To move to a new host with the data, stop the bot, copy the newest dump out with `runpodctl send` from a pod terminal (SSH or the console web terminal), create the new pod, and restore with `pg_restore --clean --if-exists -d postgres://sparky:sparky@127.0.0.1:5432/sparky`.

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

Phoenix holds one `llm` span per model call (full prompt, full reply, token counts, latency) inside a trace tree per conversation; [loupe](https://github.com/ashworks1706/loupe) reads these spans for fine-tuning. Prometheus holds server-side time series: throughput, queue depth, batching.

### Phoenix

```bash
just phoenix       # trace UI on http://localhost:6006, loopback
```

One container, `arizephoenix/phoenix:version-20.11.0`, data in the `phoenixdata` volume, UI and OTLP endpoint on one port. Export is off until `SPARKY_TELEMETRY__PHOENIX_URL=http://localhost:6006` is in `.env`. For the compose apps set `SPARKY_PHOENIX_URL=http://phoenix:6006`. Phoenix has no authentication by default and holds full prompts and replies, so in production it has no host port. When it does authenticate, set `SPARKY_TELEMETRY__PHOENIX_API_KEY`; every app sends it as a bearer token, and so does `loupe data export` in [loupe](https://github.com/ashworks1706/loupe), which reads `llm` spans through `GET /v1/projects/<project>/spans`.

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
