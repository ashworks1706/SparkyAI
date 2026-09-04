---
name: check
description: Run the full gate (`just check`: fmt-check, lint, tests, dependency direction for every unit) and fix anything it reports. Use before every commit and after finishing any code change.
---

Run `just check` from the repo root. It runs, per unit:

- Rust: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `./scripts/check-deps.sh`
- `apps/scraper`, `apps/training`, `apps/sandbox`: `ruff check`, `ruff format --check`, `pytest`
- `apps/web`: `eslint`, `vite build`

If only one unit changed, `just check-rust` / `check-scraper` / `check-training` / `check-sandbox` / `check-web` is fine.

Fix failures at the source — never `#[allow(...)]` a lint, add a `# noqa`, or skip a test to get green. If a lint is genuinely wrong for a case, the allow goes on the smallest scope possible with a one-line reason.

Report: which step failed (if any), what you changed, final status.
