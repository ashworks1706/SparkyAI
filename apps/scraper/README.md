# apps/scraper

Ingestion and live search. Fetches ASU pages, chunks and embeds them, writes the retrieval index the engine reads, and answers the engine's live queries. The engine reaches it only through the `jobs` table. Owns the schema in `migrations/`.

```bash
just scraper serve                  # the job queue: live searches, their indexing, scheduled runs
just scraper run library_hours      # one source
just scraper run --category housing # every source in a category
just scraper run --all
just scraper status                 # sources, then the queue by kind and status
just scraper login                  # capture the admin ASU session (MyASU and Duo)
just migrate                        # apply migrations/
```

`serve` and `run` require the admin session and sign in first when run at a terminal.

| Module | Holds |
|---|---|
| `ingest/fetch.py` | Firecrawl (`just crawl`) by default, main content as markdown. `SPARKY_SCRAPER__FETCHER=http` uses httpx, with headless Chromium for sources marked `needs_js` |
| `ingest/drivers/` | browser drivers: `public.py` (no session), `asu_sso.py` (MyASU sign-on), `admin.py` (the admin authenticated driver) |
| `ingest/extract.py` | HTML to text, for the `http` fetcher |
| `ingest/chunk.py` | text to chunks |
| `ingest/embed.py` | llama-server embed endpoint |
| `ingest/tree.py` | the hierarchical index: cluster, summarize on the chat endpoint, embed, recurse |
| `ingest/pipeline.py` | fetch, hash, snapshot, extract, chunk, embed, index, tree |
| `ingest/pace.py` | spaces fetches to one host `scraper.host_gap_secs` apart |
| `jobs.py` | `scraper serve`: the live and background lanes over the `jobs` queue, and the timer that queues due sources |
| `query/registry.py` | the live query sources the engine may call, published by `scraper serve` |
| `query/run.py` | one live query: checks, fetch, the text handed back |
| `query/index.py` | indexing a live result under the right source |
| `query/sources/` | live query sources, one module each |
| `sources/` | scheduled sources, one module per source with its own extractor; `pages.py` lists static pages |
| `store/` | psycopg pool and object storage; the only place a connection opens |
| `migrations/` | the schema, shared with `apps/engine` |

With the `http` fetcher, install Chromium once with `uv run playwright install chromium`.

The engine queries `chunks` with the same embedding model and dimension used here; changing the model means re-embedding every chunk. `SPARKY_SCRAPER__TREE_ENABLED=true` adds summary levels above the leaves, at a chat call and an embedding call per cluster per source.

Details: Inside scraper, Knowledge, Live source queries, and Job queue in [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
