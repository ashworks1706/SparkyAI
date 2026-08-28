# 0004 — Monorepo of services, not a single process

**Context.** "Monolithic" was initially read as one process. The intent was one repository. Discord, HTTP API, ingestion, web, and training have different runtimes, lifecycles, and scaling needs.

**Decision.** One repo; top-level folders are deployable units or shared data:
`apps/backend` (Rust, one binary run as `api` and `discord` processes), `apps/web` (static), `services/ingest` (Python worker), `services/sandbox` (Phase 7), `models` (Python, GPU), `evals` (data).

`api` is the only process that links the harness and talks to models and stores. `discord` is an HTTP/SSE client of `api`, as web and admin will be. Ingestion moved from Rust to Python because it is offline, batch, and parser-heavy — the Python scraping ecosystem and iteration loop are better, and none of Rust's advantages apply.

Services communicate only via `discord → api`, `api → vLLM / MCP / sandbox`, and the datastores. The Postgres schema (`apps/backend/crates/storage/migrations`) is the contract; Python reads it, never migrates.

**Consequences.** Two images (`backend`, `ingest`); Compose runs `api`, `discord`, `ingest`. `discord` crate no longer depends on `harness`. `chromiumoxide` and `scraper` leave the workspace. Language is never a folder; domain is never a folder.
