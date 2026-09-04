# apps/scraper

Offline ingestion. Fetches public ASU pages, chunks and embeds them, and writes the retrieval
index the engine reads. Never on the request path.

```bash
cd apps/scraper
uv sync --extra dev
uv run scraper migrate              # apply migrations/ (schema owner)
uv run scraper run library_hours    # one source
uv run scraper run --all
uv run scraper schedule             # every enabled source on its interval
uv run scraper status
```

| Module | Holds |
|---|---|
| `fetch.py` | httpx; Playwright where the page needs JS |
| `extract.py` | HTML → text |
| `chunk.py` | text → chunks |
| `embed.py` | llama-server embed endpoint |
| `pipeline.py` | fetch → hash → snapshot → extract → chunk → embed → index |
| `sources/` | one module per ASU source; a source is a row, not a folder |
| `store/` | psycopg pool, object storage; the only place a connection is opened |
| `migrations/` | the schema, shared with `apps/engine` |

Pipeline per source: fetch → content hash (skip if unchanged) → raw snapshot to object storage
→ extract → chunk → embed → write `chunks` and `source_versions`.

The engine queries `chunks` with the same embedding model and dimension used here. Changing the
model means re-embedding every chunk.
