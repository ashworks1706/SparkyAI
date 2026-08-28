# 0008 — `apps/knowledge` owns every store

**Context.** Storage adapters and query-side retrieval lived in the engine (Rust) while the scraper (Python) wrote to the same index. Chunker and embedding versions had to be kept in sync across two languages, and "who owns the data" was split.

**Decision.** One Python service, `apps/knowledge`, owns Postgres, Redis, Qdrant, and object storage. It exposes `/search`, `/memory`, `/conversations`, `/sources`, `/health` behind a service token and runs the scraper as a second process (`knowledge-scraper`) from the same package, so write and read share `index/` and `store/`. It owns the migrations. The engine holds no database connections; `engine/src/clients/knowledge.rs` implements the harness store traits over HTTP. `sqlx`, `redis`, and `qdrant-client` leave the Rust workspace.

**Cost.** One HTTP hop per store operation on the request path, a second service to run, and a wire contract (`api/schemas.py` ↔ harness types) to keep in sync. Accepted in exchange for single ownership of the data layer and one language for the whole index write/read path.

**Consequences.** Images: `sparkyai-rust`, `sparkyai-knowledge`, `sparkyai-sandbox`. Compose services `knowledge` and `scraper` from one image. Supersedes the "storage never becomes a service" line in 0006.
