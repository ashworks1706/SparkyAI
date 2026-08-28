# apps/scraper

Python worker that keeps the knowledge index fresh. Runs on a schedule, never on the request path.

```bash
cd apps/scraper
uv sync --extra dev
uv run playwright install chromium     # only for JS-rendered sources
uv run scraper run library_hours
uv run scraper run --all
uv run scraper status
```

Pipeline per source: fetch → content hash (skip if unchanged) → raw snapshot to object storage → extract → chunk → embed (vLLM embed endpoint) → index (Qdrant chunks + Postgres `source_versions`).

Reads `SPARKY_*` env like the engine. Schema is owned by `apps/engine/migrations`; this worker never migrates.
