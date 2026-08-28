# 0004 — Monorepo of services, not a single process

**Context.** "Monolithic" was initially read as one process. The intent was one repository. Discord, HTTP API, ingestion, web, and training have different runtimes, lifecycles, and scaling needs.

**Decision.** One repo. `apps/` holds every process regardless of language — `api` and `discord` (Rust bins), `ingest` (Python worker), `web` (static), `sandbox` (Phase 7). `crates/` holds Rust libraries only. `models` (Python, GPU) and `evals` (data) sit at the root.

`api` is the only process that links the harness and talks to models and stores. `discord` is an HTTP/SSE client of `api`, as web and admin will be. Ingestion moved from Rust to Python because it is offline, batch, and parser-heavy — the Python scraping ecosystem and iteration loop are better, and none of Rust's advantages apply.

Services communicate only via `discord → api`, `api → vLLM / MCP / sandbox`, and the datastores. The Postgres schema (`apps/backend/crates/storage/migrations`) is the contract; Python reads it, never migrates.

**Consequences.** Two images (`rust` with both bins, `ingest`); Compose runs `api`, `discord`, `ingest`. No process-selecting CLI: each app is its own binary with its own `main`. `crates/runtime` holds config and telemetry so both bins share them. `apps/discord` depends only on `runtime`. `chromiumoxide` and `scraper` leave the workspace. Language is never a folder; domain is never a folder.
