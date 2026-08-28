# Deploy

## Local

```bash
cp .env.example .env            # fill in Discord token and RunPod URLs
docker compose -f deploy/compose.yml up -d
```

Starts api, discord, ingest, Postgres 17, Redis 7, Qdrant, MinIO. `api` and `discord` are the same image with different commands. vLLM is not run locally; point `SPARKY_MODEL__BASE_URL` at a RunPod pod.

## RunPod

Two pods, both from the `vllm/vllm-openai` image with `deploy/runpod/start.sh` as the start command:

| Pod | Env file | GPU | Port |
|---|---|---|---|
| chat | `runpod/vllm.env` | A100 80GB / H100 | 8000 |
| embed + rerank | `runpod/embedding.env` | L4 / A10 | 8001, 8002 |

Set `SPARKY_MODEL__BASE_URL` etc. to `https://<pod-id>-<port>.proxy.runpod.net/v1`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-backend` and `sparkyai-ingest` tagged `<sha>` and `main` on every push to `main`.

## Observability

- Errors: Sentry (`SPARKY_TELEMETRY__SENTRY_DSN`)
- Traces: OTLP to Axiom (`SPARKY_TELEMETRY__OTLP_ENDPOINT`, `AXIOM_TOKEN`, `AXIOM_DATASET`)
- Logs: JSON to stdout outside development; ship with the platform's log driver
