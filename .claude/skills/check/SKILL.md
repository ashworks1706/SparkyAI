---
name: check
description: Run the full pre-commit gate (fmt, clippy -D warnings, tests) and fix anything it reports. Use before every commit and after finishing any code change.
---

Run from the repo root, in order, stopping at the first failure:

1. `cargo fmt --all`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`
4. `./scripts/check-deps.sh`

If Python packages changed (`apps/ingest`, `models`): `uvx ruff check . && uvx ruff format --check . && uv run pytest -q` in that package. If `apps/web` changed: `npm run lint && npm run build`.

Fix failures at the source — never `#[allow(...)]` a lint or skip a test to get green. If a lint is genuinely wrong for a case, the allow goes on the smallest scope possible with a one-line reason.

Report: which step failed (if any), what you changed, final status of all four.
