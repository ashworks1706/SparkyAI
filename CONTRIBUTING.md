# Contributing

SparkyAI is an open-source agent for ASU students and their orgs. Read [AGENTS.md](AGENTS.md) for
the commands, the layering rules and the conventions every change follows, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing a boundary. Both apply to people and
to coding agents alike.

## Setup

Needs [just](https://just.systems), a Rust toolchain, [uv](https://docs.astral.sh/uv), Node and
Docker. `just doctor` names anything missing and how to install it.

```
just doctor        # what is missing
just bootstrap     # .env, hooks, dependencies, datastores, schema
just check         # the gate: every unit
```

`bootstrap` is the whole first run: it copies `.env` from `.env.example`, points
`core.hooksPath` at `.githooks`, installs each unit's dependencies, starts postgres, redis and
minio, and applies the migrations. The pre-commit hook then runs the recipes for the units a
commit touches. CI runs the same recipes, so a change that touches only the scraper never waits
on a Rust build.

### A model

`bootstrap` does not give you one, and the engine answers nothing without it. Three ways:

| | |
|---|---|
| `just model` | `llama-server` on CUDA. Needs an NVIDIA GPU and the container toolkit. |
| `just model-cpu` | the same GGUFs on the processor. No GPU needed, and slow enough that you will not want it for anything but a smoke test. |
| hosted | set `SPARKY_MODEL__BASE_URL` and `SPARKY_MODEL__API_KEY` in `.env` to any OpenAI-compatible endpoint, and the same pair under `SPARKY_EMBEDDING__` for embeddings. |

Without one the engine still starts: `/health/live` answers, `/health/ready` reports
`{"postgres":true,"model":false}`, and `/chat` returns 502 `the model is unavailable`. That is
enough to work on everything except answering.

### The sandbox

`run_sandbox` needs a container runtime. With Docker running you get it for free; without one
the engine logs a warning at boot and does not offer the tool, and everything else works. Set
`SPARKY_SANDBOX__REQUIRED=true` to make a missing runtime a boot failure instead, which is what
production does.

## Branches and pull requests

Branch from `main` and name the branch for the change (`reply-threads`, `fix-cache-handoff`).
One logical change per pull request, with a test for new behaviour, and the test must fail
without the change.

Commit subjects are imperative, at most 72 characters, with no trailing period. The body explains
why; the diff already shows how.

A change is not done until `just check` passes. That is `cargo fmt --check`, `cargo clippy
--workspace --all-targets -D warnings`, `cargo test --workspace` and `scripts/check-deps.sh` for
Rust, plus ruff and pytest for the Python units and eslint, tsc and a build for the website.

Clippy runs on the CI toolchain, which is stable and moves. A lint that passes locally can fail
in CI on an older local toolchain; `rustup update stable` before blaming the diff.

### A stacked branch

CI runs on pull requests into `main`. A pull request into another branch runs no gate at all, so
run it by hand from the Actions tab: **CI** has a `workflow_dispatch` trigger, and a manual run
gates every unit rather than only the ones whose paths changed.

## Rules that are not style

- `core` imports nothing else in its crate; `engine`, `discord` and `cli` never depend on each
  other, which `scripts/check-deps.sh` enforces.
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!` or `println!`. The workspace
  lints deny them, in tests too.
- Every tunable value goes in `sparky.toml` at its default, in the same change. `.env` holds only
  secrets, per-machine URLs, and what compose and the justfile read.
- Only `apps/scraper` writes the retrieval index or fetches pages. Model output is never written
  back as evidence.
- Write-side tools go through `Policy`; consequential actions need a confirmation.
- Comments are plain ASCII and say what the code does, never why. The why belongs in the commit
  message. See Conventions in AGENTS.md.

## CI and CD

| Workflow | Runs on | Does |
|---|---|---|
| CI | pull requests into main, main, manual | the gate for each unit that changed, or every unit on a manual run |
| CD | main, tags | builds and publishes the images |

## Reporting

Bugs, wrong answers and feature requests: the issue templates. Vulnerabilities: privately, see
[SECURITY.md](SECURITY.md).
