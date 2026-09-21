# SparkyAI tasks; just alone lists them.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools, .env, and git hooks
doctor:
    ./scripts/doctor.sh

# Create .env from the example; settings live in sparky.toml
env:
    @[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo "created .env, fill in tokens and model URLs"; }

# Install the pre-commit hook (runs the gate for touched units)
hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: .githooks/pre-commit"

# Set up a fresh clone: .env, hooks, deps, datastores, schema
bootstrap: env hooks setup infra migrate
    @echo "ready. Next a model, because the engine answers nothing without one:"
    @echo "  GPU:    just model       llama-server on CUDA"
    @echo "  no GPU: just model-cpu   the same models on the processor, slowly"
    @echo "  hosted: set SPARKY_MODEL__BASE_URL and SPARKY_MODEL__API_KEY in .env"
    @echo "Then 'just cli' for the console, or 'just engine' and 'just discord' in separate shells."

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

# Rust fmt check, clippy, tests, dependency direction
check-rust:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    ./scripts/check-deps.sh

# Run the engine
engine *ARGS:
    cargo run -p engine -- {{ARGS}}

# Run the discord bot
discord *ARGS:
    cargo run -p discord -- {{ARGS}}

# Developer console; requires the admin session
cli: scraper-session
    cargo run -p cli --release

# ---------- python: scraper + training ----------

# Lint and test the scraper
check-scraper:
    cd apps/scraper && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

# Lint and test training
check-training:
    cd apps/training && uvx ruff check . && uvx ruff format --check . && uv run pytest -q

# Run a scraper command, such as serve, run library_hours, or login
scraper *ARGS:
    cd apps/scraper && uv run scraper {{ARGS}}

# Require the admin session; sign in through MyASU and Duo if it is missing
scraper-session:
    cd apps/scraper && uv run scraper login --if-needed

# Apply migrations
migrate:
    cd apps/scraper && uv run scraper migrate

# Run a training command
train *ARGS:
    cd apps/training && uv run train {{ARGS}}

# Run evals
eval *ARGS:
    cd apps/training && uv run eval {{ARGS}}

# Run a dataset command
data *ARGS:
    cd apps/training && uv run data {{ARGS}}

# ---------- web ----------

# Lint, typecheck, test, and build the web app
check-web:
    cd apps/web && npm run lint && npm run typecheck && npm test && npm run build

# Vite dev server
web:
    cd apps/web && npm run dev

# ---------- infra ----------

# Start the dev stack; requires the admin session
up *ARGS: scraper-session
    docker compose -f deploy/compose.yml up -d {{ARGS}}

# Stop the dev stack across every profile
down: sandbox-down
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile phoenix down

# Remove the sandbox containers and egress proxy the engine starts outside compose
sandbox-down:
    #!/usr/bin/env bash
    set -uo pipefail
    # DOCKER_HOST from .env selects the runtime that holds them.
    [ -f .env ] && set -a && . ./.env && set +a
    held=$(docker ps -aq --filter "label=sparky.sandbox" 2>/dev/null)
    if [ -n "$held" ]; then docker rm --force $held >/dev/null; echo "removed $(echo "$held" | wc -l) sandbox containers"; fi
    docker rm --force "${SPARKY_SANDBOX__EGRESS_PROXY_NAME:-sparky-sandbox-proxy}" >/dev/null 2>&1 || true
    docker network rm "${SPARKY_SANDBOX__EGRESS_NETWORK:-sparky-sandbox}" >/dev/null 2>&1 || true

# Start production from GHCR images
prod-up *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml pull
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml up -d {{ARGS}}

# Stop production
prod-down: sandbox-down
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml down

# Follow production logs
prod-logs *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.prod.yml logs -f {{ARGS}}

# Start postgres, redis, and minio for a host-side engine
infra *ARGS:
    docker compose -f deploy/compose.yml up -d --wait {{ARGS}} postgres redis minio

# Phoenix trace UI on localhost:6006
phoenix *ARGS:
    docker compose -f deploy/compose.yml --profile phoenix up -d {{ARGS}} phoenix

# llama-server chat and embed on the CPU
model-cpu *ARGS:
    docker compose -f deploy/compose.yml -f deploy/compose.cpu.yml --profile model up -d {{ARGS}} chat embed

# llama-server chat on :8000 and embed on :8001
model *ARGS:
    docker compose -f deploy/compose.yml --profile model up -d {{ARGS}} chat embed

# SearXNG for the web source of search_live on localhost:8888
search *ARGS:
    docker compose -f deploy/compose.yml --profile search up -d {{ARGS}} searxng

# Self-hosted Firecrawl for the scraper on :3002
crawl *ARGS:
    docker compose -f deploy/compose.yml --profile crawl up -d {{ARGS}} firecrawl

# pgweb database browser on localhost:8081
db *ARGS:
    docker compose -f deploy/compose.yml --profile db up -d {{ARGS}} pgweb

# Prometheus (:9090) and Grafana (:3000); requires SPARKY_GRAFANA_PASSWORD
metrics *ARGS:
    docker compose -f deploy/compose.yml --profile metrics up -d {{ARGS}} prometheus grafana

# GPU utilization and VRAM into Prometheus; requires NVIDIA GPU
gpu-metrics *ARGS:
    docker compose -f deploy/compose.yml --profile gpu-metrics up -d {{ARGS}} gpu-exporter

# Compose containers across every profile
ps:
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile phoenix ps -a

# Follow dev stack logs across every profile
logs *ARGS:
    docker compose -f deploy/compose.yml --profile model --profile crawl --profile db --profile metrics --profile gpu-metrics --profile phoenix logs -f {{ARGS}}

# Build the rust and scraper images locally
images:
    docker build -f deploy/docker/rust.Dockerfile -t sparkyai-rust .
    docker build -f deploy/docker/scraper.Dockerfile -t sparkyai-scraper .

# Build the sandbox and egress proxy images under the tags the engine runs
sandbox-images:
    docker build -f deploy/docker/sandbox.Dockerfile -t ghcr.io/ashworks1706/sparkyai-sandbox:main .
    docker build -f deploy/docker/sandbox-proxy.Dockerfile -t ghcr.io/ashworks1706/sparkyai-sandbox-proxy:main .

# ---------- docs ----------

# Render the ARCHITECTURE.md mermaid diagrams to verify syntax
diagrams:
    #!/usr/bin/env bash
    set -euo pipefail
    d=$(mktemp -d)
    awk -v d="$d" '/^```mermaid/{n++; f=d"/d"n".mmd"; next} /^```/{f=""; next} f{print > f}' docs/ARCHITECTURE.md
    for f in "$d"/*.mmd; do npx -y @mermaid-js/mermaid-cli -i "$f" -o "${f%.mmd}.svg" -q && echo "ok $(basename "$f")"; done
