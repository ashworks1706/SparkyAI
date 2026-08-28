# Architecture

Companion to [ROADMAP.md](ROADMAP.md). Describes the target shape; crates and traits appear as they are implemented.

## Request lifecycle

```
edge (discord | http)
  └─ RequestContext { request_id, user, permissions, budget, trace }
       └─ harness::Agent::run
            ├─ ConversationStore   load history
            ├─ Retriever           evidence for the query
            ├─ ModelProvider       generate (stream) → text | tool calls
            ├─ Policy              may this tool run for this user? confirm?
            ├─ Tool                execute → typed result | typed error
            ├─ loop until final answer, step limit, timeout, or cancel
            ├─ MemoryStore         write-policy-gated memory
            └─ TraceSink           every event, replayable
  └─ reply with citations
```

Ingestion runs as separate offline jobs and only writes to indexes. The request path never fetches external pages.

## Crates

| Crate | Owns | Depends on |
|---|---|---|
| `harness` | `RequestContext`, message types, traits below, agent loop, JSONL tracing | — |
| `model` | `ModelProvider` impls: OpenAI-compatible (vLLM), mock | harness |
| `tools` | Built-in `Tool` impls | harness |
| `retrieval` | `Retriever` impls (Qdrant), ingestion jobs, chunking, embedding client | harness |
| `discord` | serenity adapter: slash commands → `RequestContext` → agent → reply | harness, model, tools, retrieval |
| `server` | axum HTTP API, health, admin endpoints | same as discord |

Rule: `harness` depends on nothing in-repo. Adapters depend on `harness`. Nothing depends on an adapter except the binaries (`discord`, `server`).

## Traits

All in `harness`. Each has a mock for tests.

| Trait | Purpose | First impl |
|---|---|---|
| `ModelProvider` | `generate(ctx, messages, tools) -> stream of text / tool calls` | OpenAI-compatible HTTP |
| `Tool` | name, JSON schema, `call(ctx, args) -> Result<Value, ToolError>` | built-ins |
| `Retriever` | `retrieve(ctx, query, filters, limit) -> Vec<Evidence>` | Qdrant + Postgres metadata |
| `ConversationStore` | load / append messages per conversation | Postgres |
| `MemoryStore` | `remember` / `recall` / `forget`, write policy | Postgres + Qdrant |
| `Policy` | `allow(ctx, action) -> Allow \| Deny \| Confirm` | role-based |
| `TraceSink` | `emit(event)` | JSONL file |
| `Sandbox` | isolated browser / process for automation (Phase 7) | — |

## Data

| Store | Holds |
|---|---|
| PostgreSQL | users, conversations, messages, sources, source versions, memories, permissions, traces index |
| Qdrant | chunk embeddings with `source_id`, `version`, `category`, `fetched_at` payload |
| Redis | rate limits, job queue, short-lived cache |
| Object storage | raw fetched documents, model artifacts |

## Invariants

- No global mutable state. Everything per-request lives in `RequestContext`.
- Model output is never written back as retrieval evidence.
- Every write-side tool goes through `Policy`; consequential actions require confirmation.
- Every model and tool call is traced and replayable.
- Sources carry `fetched_at`; answers surface it.
