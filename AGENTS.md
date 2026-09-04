# SparkyAI — agent guide

Rust rebuild of an ASU student copilot. Read `docs/ROADMAP.md` for what we're building and in what order; `docs/ARCHITECTURE.md` for crate boundaries, traits, and invariants. Do not contradict either — propose an edit to the doc instead.

## Commands

`just` is the entrypoint for every unit (`just` lists recipes). Install: https://just.systems.

```
just doctor | env | hooks | bootstrap   # first run
just check            # fmt-check, lint, test every unit — the gate; CI and the pre-commit hook run the same recipes, only for the units a change touches
just check-rust       # cargo fmt --check, clippy -D warnings, test, scripts/check-deps.sh
just check-scraper    # ruff + pytest in apps/scraper
just check-training   # ruff + pytest in apps/training
just check-sandbox    # ruff + pytest in apps/sandbox
just check-web        # eslint + vitest + vite build in apps/web
just fmt              # format every unit in place
just setup            # install every unit's deps
just engine | discord # run a Rust app (needs .env, see .env.example)
just scraper ...      # e.g. just scraper run library_hours
just migrate
just train | eval | data ...
just infra            # postgres, redis, minio only
just up | down | logs # full compose stack (dev, builds locally)
just prod-up | prod-down | prod-logs   # GHCR images, SPARKY_IMAGE_TAG
just diagrams         # render ARCHITECTURE.md mermaid to verify syntax
```

A change is not done until `just check` passes.

## Layout

One repo. Everything that runs is under `apps/`. Language is never a folder; ASU domain is never a folder.

```
apps/engine/      Rust bin — the agent + HTTP surface. Modules: core/{config,telemetry,types,traits,tests}, agent/{harness,model,tools}, stores, routes.
apps/discord/     Rust bin — serenity bot; HTTP client of engine. Never links engine. core/{config,telemetry,types,tests}.
apps/scraper/     Python — offline ingestion: fetch, chunk, embed, write the index. Migrations live here. core/{settings,types,tests}.
apps/web/         static frontend + admin UI (Vite + React)
apps/sandbox/     Python + Playwright browser worker (Phase 7); HTTP task protocol called by engine
apps/training/    Python — datasets, post-training, eval runners + eval cases (GPU, occasional)
deploy/           compose, one Dockerfile per image, inference/ (model serving config)
docs/             ROADMAP.md, ARCHITECTURE.md, decisions/
```

Processes talk only via: discord → engine, engine → PostgreSQL / llama-server / MCP / sandbox, scraper → PostgreSQL / llama-server embed. The scraper never serves a request; it and the engine meet only in the database. `apps/scraper/migrations` is the contract.

## Dependencies we build on

- **Rig** (`rig-core`, crate name `rig_core`): the OpenAI-compatible client for chat and embeddings (`agent/model/rig_openai.rs`). Rerank is direct HTTP because Rig has no provider for llama-server's `/v1/rerank`. Never `rig::Agent` — the loop is ours.
- **rmcp**: MCP. Never hand-roll MCP.
- Everything else in the harness module (loop, policy, context assembly, memory, tracing, replay) is written here.

## Config

All settings come from `SPARKY_<SECTION>__<KEY>` env vars into `apps/engine/src/config.rs` and `apps/discord/src/config.rs` (Rust) and `settings.py` (Python packages). Secrets are `SecretString`; never log them. Add a field there and to `.env.example` in the same change.

## Rules

- Inside `apps/engine`: `core` imports nothing else in the crate; `agent::harness`, `agent::model`, `agent::tools`, and `stores` import only `core`, never each other; `routes`/`wiring` compose them. Types live in `core/types`, traits in `core/traits`; everything else is `impl` blocks and functions. Convention, checked in review. Between apps: `engine` and `discord` never depend on each other — enforced by `scripts/check-deps.sh`.
- Workspace lints are the law: no `unwrap`/`expect`/`panic`/`todo!`/`unimplemented!`/`dbg!`/`println!`, no wildcard imports, docs on every public item. Enforced by `[workspace.lints]` in `Cargo.toml`.
- A crate's public surface is its constructors and the `harness` traits it implements. Nothing reaches into another adapter.
- No global mutable state. Per-request data goes in `RequestContext`.
- Every replaceable dependency sits behind a trait in `engine/src/agent/harness` with a mock impl for tests.
- The engine reads the database; only `apps/scraper` writes the retrieval index and fetches pages.
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

- Every app has a `core/` (`src/core/` in Rust) holding what the rest of the app builds on and nothing that does work: config/settings, telemetry, every struct/enum/class, every trait, and `tests`. Rust: `core/{config,telemetry,types,traits,tests}`; Python: `core/{settings.py,types.py,tests/}`. No `struct`, `enum`, `trait`, or `class` is defined outside `core`; the other modules hold only `impl` blocks and functions. Domain code imports from `core`; `core` imports nothing from the app.
- Public items have a one-line doc comment saying what, not how.
- Commit messages: imperative subject ≤ 72 chars, body explains why.
- The tree is scaffolded ahead of code. Fill a stub in place; don't create parallel files or rename stubs without updating ARCHITECTURE.md.
- Keep docs lean. No filler prose.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`. Don't build toward those without an explicit decision.
