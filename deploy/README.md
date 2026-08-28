# Deploy

## Local

```bash
just bootstrap     # .env, git hooks, deps, datastores (postgres, redis, qdrant, minio)
just engine        # then in separate shells: just knowledge, just discord, just web
# or run everything in containers:
just up            # docker compose -f deploy/compose.yml up -d
```

Starts engine, discord, knowledge (api), scraper (same image as knowledge), Postgres 17, Redis 7, Qdrant, MinIO. The Phase 7 sandbox is behind a compose profile: `just up --profile sandbox`. `engine` and `discord` are the same image (`rust.Dockerfile`) with different entrypoints. vLLM is not run locally; point `SPARKY_MODEL__BASE_URL` at a RunPod pod.

## Production

```bash
git clone https://github.com/ashworks1706/SparkyAI.git && cd SparkyAI
cp .env.example .env               # production values; SPARKY_APP__ENV=production
SPARKY_IMAGE_TAG=main just prod-up # pulls ghcr.io images; datastores have no host ports
just prod-logs engine
```

`deploy/compose.prod.yml` overrides `compose.yml`: prebuilt images instead of builds, no host ports except engine `:8080`. Put a reverse proxy with TLS in front of engine. vLLM stays on RunPod.

## RunPod

Two pods, both from the `vllm/vllm-openai` image with `apps/inference/start.sh` as the start command:

| Pod | Env file | GPU | Port |
|---|---|---|---|
| chat | `apps/inference/vllm.env` | A100 80GB / H100 | 8000 |
| embed + rerank | `apps/inference/embedding.env` | L4 / A10 | 8001, 8002 |

Set `SPARKY_MODEL__BASE_URL` etc. to `https://<pod-id>-<port>.proxy.runpod.net/v1`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust`, `sparkyai-knowledge`, and `sparkyai-sandbox` tagged `<sha>` and `main` on every push to `main`.

## Observability

- Errors: Sentry (`SPARKY_TELEMETRY__SENTRY_DSN`)
- Traces: OTLP to Axiom (`SPARKY_TELEMETRY__OTLP_ENDPOINT`, `AXIOM_TOKEN`, `AXIOM_DATASET`)
- Logs: JSON to stdout outside development; ship with the platform's log driver
