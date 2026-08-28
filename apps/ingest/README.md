# services/ingest

Python worker that keeps the knowledge index fresh. Runs on a schedule, never on the request path.

```bash
cd apps/ingest
uv sync --extra dev
uv run playwright install chromium     # only for JS-rendered sources
uv run sparky-ingest run library_hours
uv run sparky-ingest run --all
```

Pipeline per source: fetch → content hash (skip if unchanged) → raw snapshot to object storage → extract → chunk → embed (vLLM embed endpoint) → index (Qdrant chunks + Postgres `source_versions`).

Reads `SPARKY_*` env like the backend. Schema is owned by `apps/api/migrations`; this service never migrates.
