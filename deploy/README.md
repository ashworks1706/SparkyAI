# Deploy

## Local

```bash
just bootstrap     # .env, git hooks, deps, datastores (postgres, redis, qdrant, minio)
just engine        # then in separate shells: just knowledge, just discord, just web
# or run everything in containers:
just up            # docker compose -f deploy/compose.yml up -d
```

Starts engine, discord, knowledge (api), scraper (same image as knowledge), Postgres 17, Redis 7, Qdrant, MinIO. The Phase 7 sandbox is behind a compose profile: `just up --profile sandbox`. `engine` and `discord` are the same image (`rust.Dockerfile`) with different entrypoints. The chat and embedding models are behind the `model` profile: `just model` starts Ollama and pulls them.

## Production

```bash
git clone https://github.com/ashworks1706/SparkyAI.git && cd SparkyAI
cp .env.example .env               # production values; SPARKY_APP__ENV=production
SPARKY_IMAGE_TAG=main just prod-up # pulls ghcr.io images; datastores have no host ports
just prod-logs engine
```

`deploy/compose.prod.yml` overrides `compose.yml`: prebuilt images instead of builds, no host ports except engine `:8080`. Put a reverse proxy with TLS in front of engine. Ollama runs on a GPU host; see `deploy/inference`.

## Models

One `ollama/ollama` server backs both chat and embeddings on `:11434/v1`. Locally `just model`
runs it in Compose; in deployment run the same image on a GPU host. Set
`SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` to `http://<host>:11434/v1`.
Details and the open reranker question: `deploy/inference/README.md`.

## Web

`apps/web` builds to static files: `npm run build` → `apps/web/dist`. Deploy to Vercel (root directory `apps/web`) or any static host. Not part of the Docker image.

## Images

CD builds and pushes `ghcr.io/ashworks1706/sparkyai-rust`, `sparkyai-knowledge`, and `sparkyai-sandbox` tagged `<sha>` and `main` on push to `main` — only the images whose inputs changed (`workflow_dispatch` rebuilds all). CI likewise runs only the units a change touches.

## Observability

- Errors: Sentry (`SPARKY_TELEMETRY__SENTRY_DSN`)
- Traces: OTLP to Axiom (`SPARKY_TELEMETRY__OTLP_ENDPOINT`, `AXIOM_TOKEN`, `AXIOM_DATASET`)
- Logs: JSON to stdout outside development; ship with the platform's log driver
