# 0007 — `apps/scraper` and `apps/training`

**Context.** `apps/ingest` (package `sparky_ingest`) and top-level `models/` + `evals/` named things by pipeline stage and by artifact rather than by what runs.

**Decision.** `apps/ingest` → `apps/scraper` (package and CLI `scraper`). `models/` → `apps/training` (package `training`, CLIs `data`, `train`, `eval`), with the shared eval cases moved to `apps/training/evals/cases`. Every Python unit now lives under `apps/` like everything else that runs; `training` is the one entry that runs occasionally on a GPU rather than as a service.

**Consequences.** Paths in decisions 0003–0006 that mention `services/ingest`, `apps/ingest`, `models/`, `evals/`, or `crates/` refer to layouts that no longer exist; this record is the current one. Image `sparkyai-scraper`; Compose service `scraper`; CI matrix `apps/scraper`, `apps/training`.
