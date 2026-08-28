# apps/knowledge

The service that owns every store. Search, index, memory, conversations, and sources are behind one HTTP API; the scraper is its write-side worker. Nothing else in the repo opens a database connection.

```bash
cd apps/knowledge
uv sync --extra dev
uv run knowledge migrate            # apply migrations/ (schema owner)
uv run knowledge-api                # :8081, service-token auth
uv run knowledge-scraper run --all  # offline; see SCRAPER.md
```

| Package | Responsibility |
|---|---|
| `api/` | `/search`, `/memory`, `/conversations`, `/sources`, `/health`; called by the engine |
| `index/` | embed, rerank, dense (Qdrant), lexical (BM25), hybrid fusion — used by both scraper and search |
| `memory/` | kinds, write policy, tenant+user scoping |
| `store/` | postgres (+ migrations), redis, qdrant, object storage clients |
| `scraper/` | fetch → snapshot → extract → chunk → embed → index; one module per ASU source |
| `migrations/` | the schema |

Two processes from one image: `knowledge-api` (request path, called by engine) and `knowledge-scraper` (batch). They share `index/` and `store/` in-process, which is the reason they live in one package: the chunker and embedding used to write are the ones used to read.

Contract with the engine: `api/schemas.py` ↔ `engine/src/agent/harness/{evidence,memory,conversation}.rs`.
