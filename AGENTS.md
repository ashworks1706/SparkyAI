# SparkyAI — agent guide

Rust rebuild of an ASU student copilot. Read `docs/ROADMAP.md` for what we're building and in what order; `docs/ARCHITECTURE.md` for crate boundaries, traits, and invariants. Do not contradict either — propose an edit to the doc instead.

## Commands

```
cargo test --workspace                                  # tests
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
cargo fmt --all                                         # before commit
```

CI runs exactly those three. A change is not done until all pass.

## Layout

```
crates/<name>/   one crate per row in ARCHITECTURE.md; create only when it has code
docs/            ROADMAP.md, ARCHITECTURE.md
models/          Python post-training (Phase 6, not yet)
```

## Rules

- `harness` depends on nothing in-repo. Adapters depend on `harness`. Only binaries depend on adapters.
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

## Conventions

- Tests live next to the code (`#[cfg(test)] mod tests`); integration tests in `crates/<name>/tests/`.
- Public items have a one-line doc comment saying what, not how.
- Commit messages: imperative subject ≤ 72 chars, body explains why.
- Don't scaffold empty modules, placeholder files, or crates ahead of need.
- Keep docs lean. No filler prose.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`. Don't build toward those without an explicit decision.
