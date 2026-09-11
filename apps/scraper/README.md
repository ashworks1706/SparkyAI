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
| `ingest/fetch.py` | Firecrawl (`just crawl`) by default: JS rendered, main content as markdown. `SPARKY_SCRAPER__FETCHER=http` uses httpx + Playwright instead |
| `ingest/extract.py` | HTML → text, for the `http` fetcher |
| `ingest/chunk.py` | text → chunks |
| `ingest/embed.py` | llama-server embed endpoint |
| `ingest/tree.py` | the hierarchical index: cluster a level, summarize each cluster on the chat endpoint, embed the summary, recurse |
| `ingest/pipeline.py` | fetch → hash → snapshot → extract → chunk → embed → index → tree |
| `query/registry.py` | the live query sources the engine may call, published by `scraper worker` |
| `query/worker.py` | claims `jobs` rows the engine queues, fetches, and writes the result |
| `sources/` | one module per ASU source; a source is a row, not a folder |
| `store/` | psycopg pool, object storage; the only place a connection is opened |
| `migrations/` | the schema, shared with `apps/engine` |

With the `http` fetcher, sources marked `needs_js` render in headless Chromium; install it once
with `uv run playwright install chromium`. Firecrawl renders everything itself.

Pipeline per source: fetch → content hash (skip if unchanged) → raw snapshot to object storage
→ extract → chunk → embed → write `chunks` and `source_versions`.

The engine queries `chunks` with the same embedding model and dimension used here. Changing the
model means re-embedding every chunk.

`SPARKY_SCRAPER__TREE_ENABLED=true` adds the levels above the leaves. A run clusters its own
chunks, summarizes each cluster with one call to `[summary]` (the chat model), embeds the
summary, and repeats to `tree_max_level` or until a level splits into fewer than two clusters.
Summaries land in `chunks` beside the leaves with `level > 0` and `parent_id` set on what they
cover, so one retrieval searches every level at once. It costs a chat call and an embedding call
per cluster per source.
