# SparkyAI monorepo tasks. `just` lists them; `just <recipe>`.
# Units: engine, discord (Rust) · knowledge, training, sandbox (Python) · web (TypeScript) · infra (Compose)

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools, .env, and git hooks
doctor:
    ./scripts/doctor.sh

# Create .env from the example (no-op if it exists)
env:
    @[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo "created .env — fill in tokens and RunPod URLs"; }

# Install the pre-commit hook (runs the gate for touched units)
hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: .githooks/pre-commit"

# Everything a fresh clone needs: tools check, .env, hooks, deps, datastores
bootstrap: env hooks setup infra
    @echo "ready: run 'just engine' / 'just knowledge' / 'just discord' in separate shells, or 'just up' for everything in docker"

# ---------- everything ----------

# Format, lint, and test every unit
check: check-rust check-knowledge check-training check-sandbox check-web
    @echo "all units ok"

# Format every unit in place
fmt:
    cargo fmt --all
    cd apps/knowledge && uvx ruff format . && uvx ruff check --fix .
    cd apps/training && uvx ruff format . && uvx ruff check --fix .
    cd apps/sandbox  && uvx ruff format . && uvx ruff check --fix .
    cd apps/web      && npx eslint . --fix

# Install every unit's dependencies
setup:
    cd apps/knowledge && uv sync --extra dev
    cd apps/training && uv sync --extra dev
    cd apps/sandbox  && uv sync --extra dev
    cd apps/web      && npm ci
    cargo fetch

# Remove build artifacts and virtualenvs
clean:
    cargo clean
    rm -rf apps/knowledge/.venv apps/training/.venv apps/sandbox/.venv apps/web/node_modules apps/web/dist

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

# ---------- python: knowledge + training + sandbox ----------

check-knowledge:
    cd apps/knowledge && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

check-training:
    cd apps/training && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

check-sandbox:
    cd apps/sandbox && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

# Sandbox worker CLI: just sandbox serve
sandbox *ARGS:
    cd apps/sandbox && uv run sandbox {{ARGS}}

# Knowledge API server (needs .env)
knowledge *ARGS:
    cd apps/knowledge && uv run knowledge-api {{ARGS}}

# Scraper worker: just scraper run library_hours
scraper *ARGS:
    cd apps/knowledge && uv run knowledge-scraper {{ARGS}}

# Apply migrations
migrate:
    cd apps/knowledge && uv run knowledge migrate

# Training CLIs: just train configs · just eval suites · just data stats
train *ARGS:
    cd apps/training && uv run train {{ARGS}}

eval *ARGS:
    cd apps/training && uv run eval {{ARGS}}

data *ARGS:
    cd apps/training && uv run data {{ARGS}}

# ---------- web ----------

check-web:
    cd apps/web && npm run lint && npm test && npm run build

# Vite dev server
web:
    cd apps/web && npm run dev

# ---------- infra ----------

# Start engine, discord, knowledge, scraper, postgres, redis, qdrant, minio
up *ARGS:
    docker compose -f deploy/compose.yml up -d {{ARGS}}

down:
    docker compose -f deploy/compose.yml down

# Production: prebuilt GHCR images (SPARKY_IMAGE_TAG=main|<sha>), no host ports for datastores
prod-up *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml pull
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml up -d {{ARGS}}

prod-down:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml down

prod-logs *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml logs -f {{ARGS}}

# Datastores + knowledge-api. Editing apps/knowledge needs `just infra --build`.
infra *ARGS:
    docker compose -f deploy/compose.yml up -d {{ARGS}} postgres redis qdrant minio knowledge

# Local vLLM on the host GPU. Override with SPARKY_LOCAL_MODEL.
model *ARGS:
    docker compose -f deploy/compose.yml --profile model up -d vllm {{ARGS}}

logs *ARGS:
    docker compose -f deploy/compose.yml logs -f {{ARGS}}

# Build both images locally
images:
    docker build -f deploy/docker/rust.Dockerfile -t sparkyai-rust .
    docker build -f deploy/docker/knowledge.Dockerfile -t sparkyai-knowledge .
    docker build -f deploy/docker/sandbox.Dockerfile -t sparkyai-sandbox .

# ---------- docs ----------

# Render every mermaid diagram in ARCHITECTURE.md to verify syntax
diagrams:
    #!/usr/bin/env bash
    set -euo pipefail
    d=$(mktemp -d)
    awk -v d="$d" '/^```mermaid/{n++; f=d"/d"n".mmd"; next} /^```/{f=""; next} f{print > f}' docs/ARCHITECTURE.md
    for f in "$d"/*.mmd; do npx -y @mermaid-js/mermaid-cli -i "$f" -o "${f%.mmd}.svg" -q && echo "ok $(basename "$f")"; done
