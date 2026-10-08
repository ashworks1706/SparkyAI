# syntax=docker/dockerfile:1.7
# All of Sparky in one RunPod GPU pod: models, engine, bot, scraper, datastores, search.
# RunPod pods cannot run containers, so supervisord runs every service as a process and
# run_sandbox stays off. State lives under /workspace, the pod volume.
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
ARG SEARXNG_REF=d4f00d15d
ARG MINIO_REF=RELEASE.2025-10-15T17-29-55Z

FROM rust:1.95-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release -p engine -p discord

# MinIO publishes no binaries or public images; this builds its last release from source.
FROM golang:1.24-bookworm AS minio
ARG MINIO_REF
RUN git clone --depth 1 --branch ${MINIO_REF} https://github.com/minio/minio.git /src \
    && cd /src && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /minio .

# llama.cpp server image: Ubuntu 24.04 with the CUDA runtime and /app/llama-server.
FROM ${LLAMA_IMAGE}
ARG SEARXNG_REF
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates curl git python3 python3-venv supervisor redis-server \
        postgresql-16 postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
COPY --from=minio /minio /usr/local/bin/minio
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy

# SearXNG, the web source of search_live, pinned to the compose image revision.
RUN git clone https://github.com/searxng/searxng.git /opt/searxng \
    && git -C /opt/searxng checkout ${SEARXNG_REF} \
    && uv venv /opt/searxng/.venv \
    && VIRTUAL_ENV=/opt/searxng/.venv uv pip install -r /opt/searxng/requirements.txt
COPY deploy/search/settings.yml /etc/searxng/settings.yml

# The scraper, with Chromium for sources that need a browser.
WORKDIR /opt/scraper
COPY apps/scraper/pyproject.toml apps/scraper/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/scraper .
RUN uv sync --no-dev \
    && .venv/bin/playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/engine /usr/local/bin/engine
COPY --from=builder /app/target/release/discord /usr/local/bin/discord
COPY sparky.toml /etc/sparky/sparky.toml
COPY deploy/runpod/supervisord.conf /etc/sparky/supervisord.conf
COPY deploy/runpod/start.sh /usr/local/bin/sparky-start
COPY deploy/runpod/backup.sh /usr/local/bin/sparky-backup

# Addresses inside the pod. Secrets come from the pod environment.
ENV SPARKY_CONFIG_FILE=/etc/sparky/sparky.toml \
    SPARKY_DATA_DIR=/workspace \
    LLAMA_CACHE=/workspace/models \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    SPARKY_APP__HTTP_ADDR=127.0.0.1:8080 \
    SPARKY_ENGINE__BASE_URL=http://127.0.0.1:8080 \
    SPARKY_POSTGRES__URL=postgres://sparky:sparky@127.0.0.1:5432/sparky \
    SPARKY_REDIS__URL=redis://127.0.0.1:6379 \
    SPARKY_MODEL__BASE_URL=http://127.0.0.1:8000/v1 \
    SPARKY_SUMMARY__BASE_URL=http://127.0.0.1:8000/v1 \
    SPARKY_EMBEDDING__BASE_URL=http://127.0.0.1:8001/v1 \
    SPARKY_MODEL__API_KEY=local \
    SPARKY_SUMMARY__API_KEY=local \
    SPARKY_EMBEDDING__API_KEY=local \
    SPARKY_OBJECT_STORE__ENDPOINT=http://127.0.0.1:9000 \
    SPARKY_OBJECT_STORE__ACCESS_KEY=minioadmin \
    SPARKY_OBJECT_STORE__SECRET_KEY=minioadmin \
    SPARKY_SEARCH__BASE_URL=http://127.0.0.1:8888 \
    SPARKY_SCRAPER__FETCHER=http \
    SPARKY_SANDBOX__ENABLED=false \
    SPARKY_SANDBOX__REQUIRED=false \
    SPARKY_AUTH__STORAGE_STATE_PATH=/workspace/auth/admin_state.json \
    SPARKY_CHAT_GGUF=Qwen/Qwen3-4B-GGUF:Q4_K_M \
    SPARKY_CHAT_CTX=16384 \
    SPARKY_CHAT_PARALLEL=2 \
    SPARKY_EMBED_GGUF=Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
    SPARKY_EMBED_CTX=4096 \
    SPARKY_EMBED_PARALLEL=2 \
    SPARKY_BACKUP_KEEP=7 \
    SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
    SEARXNG_SECRET=change-me-in-the-pod-environment
WORKDIR /
ENTRYPOINT ["/usr/local/bin/sparky-start"]
