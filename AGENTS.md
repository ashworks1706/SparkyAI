# SparkyAI — agent guide

Rust rebuild of an ASU student copilot. Read `docs/ROADMAP.md` for what we're building and in what order; `docs/ARCHITECTURE.md` for crate boundaries, traits, and invariants. Do not contradict either — propose an edit to the doc instead.

## Commands

```
cargo test --workspace                                  # tests
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
cargo fmt --all                                         # before commit
cargo run -p sparky-app -- serve|ingest|migrate         # needs .env (see .env.example)
docker compose -f deploy/docker-compose.yml up -d       # postgres, redis, qdrant, minio
cd apps/web && npm ci && npm run lint && npm run build   # web
```

Plus `./scripts/check-deps.sh`, which fails if a crate imports something it shouldn't. CI runs all four. A change is not done until all pass.

## Layout

```
crates/<name>/   one crate per row in ARCHITECTURE.md; module files are pre-scaffolded with a doc comment
apps/web/        landing site + future admin UI; Vite + React; static, no shared code with crates/
docs/            ROADMAP.md, ARCHITECTURE.md
models/          Python post-training (Phase 6, not yet)
```

## Dependencies we build on

- **Rig** (`rig-core`): model clients, `Tool` schema, embeddings, vector stores. Never `rig::Agent` — the loop is ours.
- **rmcp**: MCP. Never hand-roll MCP.
- Everything else in the harness (loop, policy, context assembly, memory, tracing, replay) is written here.

## Config

All settings come from `SPARKY_<SECTION>__<KEY>` env vars into `crates/app/src/config.rs`. Secrets are `SecretString`; never log them. Add a field there and to `.env.example` in the same change.

## Rules

- One binary (`sparky-app`). `harness` depends on nothing in-repo; adapters depend only on `harness`; only `app` depends on adapters. Enforced by `scripts/check-deps.sh`.
- Workspace lints are the law: no `unwrap`/`expect`/`panic`/`todo!`/`unimplemented!`/`dbg!`/`println!`, no wildcard imports, docs on every public item. Enforced by `[workspace.lints]` in `Cargo.toml`.
- A crate's public surface is its constructors and the `harness` traits it implements. Nothing reaches into another adapter.
- No global mutable state. Per-request data goes in `RequestContext`.
- Every replaceable dependency sits behind a trait in `harness` with a mock impl for tests.
- The request path never makes external HTTP calls except to the model server and our own stores. Fetching pages is ingestion.
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
