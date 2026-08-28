---
name: new-crate
description: Create a new workspace crate following repo conventions. Use only when docs/ARCHITECTURE.md gains a new crate; existing crates are already scaffolded.
---

Arguments: crate name (e.g. `model`, `retrieval`).

1. Confirm the crate is listed in `docs/ARCHITECTURE.md`. If not, stop and ask — new crates are an architecture decision.
3. Create `crates/<name>/Cargo.toml`:
   - package name `sparky-<name>`
   - `version/edition/license/repository` all `.workspace = true`
   - one-line `description`
   - deps via `.workspace = true`; add new ones to root `[workspace.dependencies]` first
   - depend on `sparky-harness = { path = "../harness" }` unless this *is* harness
4. `src/lib.rs` with a `//!` crate doc referencing `docs/ARCHITECTURE.md`.
5. Run `/check`.
6. Update the crate table in `docs/ARCHITECTURE.md` only if the crate's responsibility differs from what's written.
