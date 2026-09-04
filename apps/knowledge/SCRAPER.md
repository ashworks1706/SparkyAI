# scraper (write side of apps/knowledge)

Python worker that keeps the knowledge index fresh. Runs on a schedule, never on the request path.

```bash
cd apps/knowledge
uv sync --extra dev
uv run playwright install chromium     # only for JS-rendered sources
uv run knowledge-scraper run library_hours
uv run knowledge-scraper run --all
uv run knowledge-scraper status
```

Pipeline per source: fetch → content hash (skip if unchanged) → raw snapshot to object storage → extract → chunk → embed (llama-server embed endpoint) → index (Postgres `chunks` + `source_versions`).

Runs from the same package and image as `knowledge-api`; shares `index/` and `store/`. Migrations are applied by `knowledge migrate`, not by the scraper.
