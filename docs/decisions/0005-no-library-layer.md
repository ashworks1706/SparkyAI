# 0005 — No library layer; the harness is a module of the API

**Context.** `crates/` existed to make the harness a separately releasable library and to let CI enforce boundaries between harness, adapters, and binaries. The harness is not a product. The product is SparkyAI.

**Decision.** Fold `crates/{harness,model,tools,retrieval,storage,runtime}` into `apps/api/src/` as modules. `apps/discord` carries its own small config and telemetry. `deploy/runpod` becomes `apps/inference` so everything that runs is under `apps/`. Binaries are named `api` and `discord`; there is no process-selecting CLI.

**Consequences.** The intra-API boundary (harness imports nothing; adapters import only harness) is a convention checked by review, not by the compiler. The inter-app rule — `discord` never links `api` — stays enforced by `scripts/check-deps.sh`. Faster to move code while the design settles; a library can be extracted later if a second consumer ever appears.

**Amended 2026-08-29.** `apps/inference` moved back to `deploy/inference`: it is RunPod deployment config (env files + start script), not something we build or run from the repo, so `apps/` stays reserved for services we write.
