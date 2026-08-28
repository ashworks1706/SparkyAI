# 0006 — `engine`, and the three seams inside it

**Context.** After 0005 the backend was `apps/api` with flat modules. The natural seams — the thing that thinks, the thing that finds, the thing that persists — were not visible in the tree, and "api" named the transport rather than the thing.

**Decision.** Rename `apps/api` → `apps/engine` (binary `engine`, config section `SPARKY_ENGINE__*`). Inside: `agent/` (harness, model, tools), `knowledge/` (query-side retrieval), `storage/` (adapters), `routes/` (HTTP). Dependency direction: `routes`/`wiring` → all; `agent::model`, `agent::tools`, `knowledge`, `storage` → `agent::harness` only; `agent::harness` → nothing.

**Not decided.** Splitting `knowledge` or `storage` into their own processes. `knowledge` becomes a service only when a second consumer or independent scaling need appears. `storage` never does — Postgres, Qdrant, and Redis already are services. Engine and HTTP never split; HTTP is the engine's surface.
