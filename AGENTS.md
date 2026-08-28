# SparkyAI — agent guide

Rust rebuild of an ASU student copilot. Read `docs/ROADMAP.md` for what we're building and in what order; `docs/ARCHITECTURE.md` for crate boundaries, traits, and invariants. Do not contradict either — propose an edit to the doc instead.

## Commands

```
cargo test --workspace                                  # Rust: crates/ + apps/api + apps/discord
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
cargo fmt --all                                         # before commit
./scripts/check-deps.sh                                 # dependency direction
cargo run -p sparky-api                                 # needs .env (see .env.example)
cargo run -p sparky-discord

cd apps/ingest && uv sync --extra dev && uv run pytest -q
cd models      && uv sync --extra dev && uv run pytest -q
cd apps/web    && npm ci && npm run lint && npm run build
docker compose -f deploy/compose.yml up -d              # api, discord, ingest, postgres, redis, qdrant, minio
```

CI runs all of these. A change is not done until the relevant ones pass.

## Layout

One repo. `apps/` holds every process; `crates/` holds Rust libraries. Language is never a folder; ASU domain is never a folder.

```
apps/api/         Rust bin: axum + harness wiring. The only process that links the harness and talks to models/stores.
apps/discord/     Rust bin: serenity bot; HTTP/SSE client of api. Never links the harness.
apps/ingest/      Python worker: scrape → chunk → embed → index
apps/web/         static frontend + admin UI (Vite + React)
apps/sandbox/     Phase 7 browser worker
crates/harness    types, traits, agent loop, tracing
crates/runtime    config + telemetry bootstrap shared by Rust apps
crates/{model,tools,retrieval,storage}   adapters; each depends only on harness
models/           Python: datasets, training, eval runners
evals/            shared eval data only
deploy/           compose, one Dockerfile per image, runpod
docs/             ROADMAP.md, ARCHITECTURE.md, decisions/
```

Processes talk only via: discord → api (HTTP), api → vLLM / MCP / sandbox. Everything else goes through Postgres, Qdrant, Redis, object storage. `crates/storage/migrations` is the contract.

## Dependencies we build on

- **Rig** (`rig-core`): model clients, `Tool` schema, embeddings, vector stores. Never `rig::Agent` — the loop is ours.
- **rmcp**: MCP. Never hand-roll MCP.
- Everything else in the harness (loop, policy, context assembly, memory, tracing, replay) is written here.

## Config

All settings come from `SPARKY_<SECTION>__<KEY>` env vars into `crates/runtime/src/config.rs` (Rust) and `settings.py` (Python packages). Secrets are `SecretString`; never log them. Add a field there and to `.env.example` in the same change.

## Rules

- `harness` and `runtime` depend on nothing in-repo. Other `crates/*` depend only on `harness`. `apps/api` may use any crate. `apps/discord` may use only `runtime`. Enforced by `scripts/check-deps.sh`.
- Workspace lints are the law: no `unwrap`/`expect`/`panic`/`todo!`/`unimplemented!`/`dbg!`/`println!`, no wildcard imports, docs on every public item. Enforced by `[workspace.lints]` in `Cargo.toml`.
- A crate's public surface is its constructors and the `harness` traits it implements. Nothing reaches into another adapter.
- No global mutable state. Per-request data goes in `RequestContext`.
- Every replaceable dependency sits behind a trait in `harness` with a mock impl for tests.
- The request path never makes external HTTP calls except to the model server, MCP servers, and our own stores. Fetching pages is `apps/ingest`.
- Model output is never written back as retrieval evidence.
- Write-side tools go through `Policy`; consequential actions require confirmation.
- Errors: `thiserror` enums per crate, no `anyhow` in library crates, no `unwrap` outside tests.
- Async: tokio. Traits use `async_trait` until native async traits cover our needs.
- Logging: `tracing` macros, structured fields, no `println!`.
- Dependencies are declared in `[workspace.dependencies]` and referenced with `.workspace = true`.
- Edition 2024. Follow rustfmt defaults; clippy warnings are errors.

## Skills

`.claude/skills/README.md` lists them. Use `rust-skills` when writing Rust, `postgres-strict` for schema/migrations, `test-driven-development` for features and fixes, `systematic-debugging` for bugs, `verification-before-completion` before saying anything is done, `security-audit-standard` before a release, `/code-quality` for cleanup and refactor passes. `/check` is the gate.

## Conventions

- Tests live next to the code (`#[cfg(test)] mod tests`); integration tests in `crates/<name>/tests/`.
- Public items have a one-line doc comment saying what, not how.
- Commit messages: imperative subject ≤ 72 chars, body explains why.
- The tree is scaffolded ahead of code. Fill a stub in place; don't create parallel files or rename stubs without updating ARCHITECTURE.md.
- Keep docs lean. No filler prose.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`. Don't build toward those without an explicit decision.
