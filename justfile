# SparkyAI monorepo tasks. Running just with no arguments lists them.
# Units: engine, discord, cli (Rust); scraper, training (Python); web (TypeScript); infra (Compose)

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools, .env, and git hooks
doctor:
    ./scripts/doctor.sh

# Create .env from the example (no-op if it exists). Settings live in the committed sparky.toml
env:
    @[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo "created .env, fill in tokens and model URLs"; }

# Install the pre-commit hook (runs the gate for touched units)
hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: .githooks/pre-commit"

# Everything a fresh clone needs: tools check, .env, hooks, deps, datastores
bootstrap: env hooks setup infra
    @echo "ready: 'just cli' for the console, or 'just engine' / 'just discord' in separate shells, or 'just up' for everything in docker"

# ---------- everything ----------

# Format, lint, and test every unit
check: check-rust check-scraper check-training check-web
    @echo "all units ok"

# Format every unit in place
fmt:
    cargo fmt --all
    cd apps/scraper && uvx ruff format . && uvx ruff check --fix .
    cd apps/training && uvx ruff format . && uvx ruff check --fix .
    cd apps/web      && npx eslint . --fix

# Install every unit's dependencies
setup:
    cd apps/scraper && uv sync --extra dev
    cd apps/training && uv sync --extra dev
    cd apps/web      && npm ci
    cargo fetch

# Remove build artifacts and virtualenvs
clean:
    cargo clean
    rm -rf apps/scraper/.venv apps/training/.venv apps/web/node_modules apps/web/dist

# ---------- rust: engine + discord ----------

# fmt --check, clippy -D warnings, tests, dependency direction
check-rust:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    ./scripts/check-deps.sh

# Run the engine (needs .env)
engine *ARGS:
    cargo run -p engine -- {{ARGS}}

# Run the discord bot (needs .env)
discord *ARGS:
    cargo run -p discord -- {{ARGS}}

# Developer console: start/stop every unit, tail logs, run tasks
cli:
    cargo run -p cli --release

# ---------- python: scraper + training ----------

check-scraper:
    cd apps/scraper && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

check-training:
    cd apps/training && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

# The scraper: just scraper serve (live searches, their indexing, scheduled runs), just scraper run library_hours
scraper *ARGS:
    cd apps/scraper && uv run scraper {{ARGS}}

# Apply migrations
migrate:
    cd apps/scraper && uv run scraper migrate

# Training CLIs: just data export|verify|stats; just eval run|baseline|compare; just train sft [--dry-run]
train *ARGS:
    cd apps/training && uv run train {{ARGS}}

eval *ARGS:
    cd apps/training && uv run eval {{ARGS}}

data *ARGS:
    cd apps/training && uv run data {{ARGS}}

# ---------- web ----------

check-web:
    cd apps/web && npm run lint && npm run typecheck && npm test && npm run build

# Vite dev server
web:
    cd apps/web && npm run dev

# ---------- infra ----------

# Start engine, discord, scraper, postgres, redis, minio
up *ARGS:
    docker compose -f deploy/compose.yml up -d {{ARGS}}

down:
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile posthog --profile phoenix down

# Production: prebuilt GHCR images (SPARKY_IMAGE_TAG=main|<sha>), no host ports for datastores
prod-up *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml pull
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml up -d {{ARGS}}

prod-down:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml down

prod-logs *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml logs -f {{ARGS}}

# Datastores: postgres, redis, minio. The engine runs on the host. PostHog is just posthog
infra *ARGS:
    docker compose -f deploy/compose.yml up -d {{ARGS}} postgres redis minio

# PostHog (traces and product events) on http://localhost:8010, loopback. 17 containers
posthog *ARGS:
    ./scripts/posthog.sh
    docker compose -f deploy/compose.yml --profile posthog up -d --no-build {{ARGS}} $(docker compose -f deploy/compose.yml --profile posthog config --services | grep -E '^posthog(-|$)')

# Phoenix trace UI on http://localhost:6006, loopback. Set SPARKY_TELEMETRY__PHOENIX_URL to export.
phoenix *ARGS:
    docker compose -f deploy/compose.yml --profile phoenix up -d {{ARGS}} phoenix

# llama-server for chat (:8000) and embeddings (:8001). GGUFs download on first run.
model *ARGS:
    docker compose -f deploy/compose.yml --profile model up -d {{ARGS}} chat embed

# SearXNG (self-hosted metasearch) for the search_web tool: http://localhost:8888, loopback
search *ARGS:
    docker compose -f deploy/compose.yml --profile search up -d {{ARGS}} searxng

# Firecrawl (self-hosted) for the scraper: API on :3002
crawl *ARGS:
    docker compose -f deploy/compose.yml --profile crawl up -d {{ARGS}} firecrawl

# Browse the database at http://localhost:8081 (pgweb, loopback)
db *ARGS:
    docker compose -f deploy/compose.yml --profile db up -d {{ARGS}} pgweb

# Prometheus (:9090) and Grafana (:3000, dashboard "SparkyAI inference"). Needs SPARKY_GRAFANA_PASSWORD.
metrics *ARGS:
    docker compose -f deploy/compose.yml --profile metrics up -d {{ARGS}} prometheus grafana

# GPU utilisation and VRAM into Prometheus. Only on a host with an NVIDIA GPU.
gpu-metrics *ARGS:
    docker compose -f deploy/compose.yml --profile gpu-metrics up -d {{ARGS}} gpu-exporter

# What's running, across every profile
ps:
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile posthog --profile phoenix ps -a

logs *ARGS:
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile posthog --profile phoenix logs -f {{ARGS}}

# Build both images locally
images:
    docker build -f deploy/docker/rust.Dockerfile -t sparkyai-rust .
    docker build -f deploy/docker/scraper.Dockerfile -t sparkyai-scraper .

# ---------- docs ----------

# Render every mermaid diagram in ARCHITECTURE.md to verify syntax
diagrams:
    #!/usr/bin/env bash
    set -euo pipefail
    d=$(mktemp -d)
    awk -v d="$d" '/^```mermaid/{n++; f=d"/d"n".mmd"; next} /^```/{f=""; next} f{print > f}' docs/ARCHITECTURE.md
    for f in "$d"/*.mmd; do npx -y @mermaid-js/mermaid-cli -i "$f" -o "${f%.mmd}.svg" -q && echo "ok $(basename "$f")"; done
