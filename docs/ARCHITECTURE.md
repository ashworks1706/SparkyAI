# Architecture

SparkyAI is a Discord copilot for the AI Society at ASU. It answers questions from public ASU sources, keeps conversation and user memory, and performs moderator actions in Discord. Later phases add MCP tools and a sandboxed browser for authenticated tasks.

This document is the target shape. Order of work is in [ROADMAP.md](ROADMAP.md); decisions are in [decisions/](decisions/).

## Stack

| Layer | Choice | Where |
|---|---|---|
| Engine, Discord bot | Rust 2024 — tokio, axum, serenity, serde, thiserror, figment | `apps/engine`, `apps/discord` |
| Model client, tool schema, embeddings | Rig (`rig-core`); our loop drives its `CompletionModel`, never `rig::Agent` | `apps/engine/src/agent` |
| MCP | `rmcp` (official SDK) | `apps/engine` (Phase 4) |
| Knowledge service | Python 3.12 — FastAPI, psycopg, qdrant-client, redis, boto3 | `apps/knowledge` |
| Scraper | httpx + BeautifulSoup; Playwright where JS is required | `apps/knowledge` (`knowledge-scraper`) |
| Sandboxed browser | Python + Playwright, FastAPI task protocol | `apps/sandbox` (Phase 7) |
| Web | Vite + React + TypeScript + shadcn | `apps/web` |
| Post-training | TRL + PEFT (+ Unsloth), W&B, HF Hub | `apps/training` |
| Evals | Inspect AI, BFCL, lm-eval | `apps/training/evals` |
| Chat model | Qwen3 on Ollama (OpenAI-compatible HTTP) | `deploy/inference` |
| Embeddings | Qwen3-Embedding-0.6B (1024-dim) on Ollama | `deploy/inference` |
| Database | PostgreSQL 17 — source of truth | reached only via `apps/knowledge` |
| Vector store | Qdrant (rebuildable; Piramid later) | reached only via `apps/knowledge` |
| Cache, queue | Redis 7 | reached only via `apps/knowledge` |
| Object storage | S3-compatible (MinIO locally) — snapshots, artifacts | reached only via `apps/knowledge` |
| Observability | Sentry (errors), OpenTelemetry → Axiom (traces), JSON logs | every app |
| Config | `SPARKY_<SECTION>__<KEY>` env vars; secrets never logged | `config.rs`, `settings.py`, `.env.example` |
| Build, gate | `just` recipes; the same ones run in the pre-commit hook and CI | `justfile`, `.githooks`, `.github/workflows` |
| Deploy | Docker Compose (dev builds locally, prod pulls GHCR); Ollama on a GPU host | `deploy/` |

## Rules

- Open models only, served by Ollama behind an OpenAI-compatible HTTP API.
- Facts come from retrieval or live observation, never from model weights.
- Public sites are ingested offline. The request path never fetches a web page.
- Only `apps/knowledge` opens a database connection. Everything else reaches stores through its HTTP API.
- Every request carries its own `RequestContext`. No global mutable state.
- Every replaceable dependency is a trait in `engine/src/agent/harness` with a mock for tests.
- Rig supplies model clients, tool schema, embeddings, and vector-store adapters. The harness owns the loop, policy, context, memory, and tracing.
- Model output is never written back as retrieval evidence.
- Anything that creates, changes, submits, posts, books, or deletes requires confirmation immediately before the action.
- Credentials, cookies, and authenticated page content never enter retrieval indexes, memory, or traces.

## Layout

```
apps/
  engine/         Rust bin. The agent and its HTTP surface. No database connections.
    src/agent/      harness (types, traits, loop, context, tracing) · model (Rig → Ollama, mock) · tools
    src/clients/    HTTP client for knowledge, implementing the harness store traits
    src/routes/     chat, health, admin
    src/{config,telemetry,wiring}.rs
  discord/        Rust bin. serenity bot; HTTP/SSE client of engine. Never links it.
  knowledge/      Python. Owns every store. Two processes from one package:
    knowledge-api      /search /memory /conversations /sources — called by engine
    knowledge-scraper  fetch → snapshot → extract → chunk → embed → index, offline
    api · index (embed, rerank, dense, lexical, hybrid) · memory · store · scraper
    migrations/        the schema
  training/       Python. datasets, post-training, eval runners; evals/cases holds the shared eval data
  sandbox/        Python + Playwright worker (Phase 7); HTTP task protocol; one context per user session
  web/            Vite + React frontend and admin UI
deploy/           compose (dev + prod), one Dockerfile per image, inference/ (model serving config)
docs/             ROADMAP.md, this file, decisions/
```

Each Python app directory is itself the importable package — `apps/knowledge` is `knowledge` — with no `src/` layer and no repeated directory name. `pyproject.toml` maps the package to `.` and lists its subpackages, so a new subpackage must be added there.

Everything that runs is under `apps/`. Language is never a folder. ASU domain (library, events, …) is never a folder either — it is a row in `sources` or an entry in a registry.

Services talk only at these edges: `discord → engine`, `engine → knowledge`, `engine → Ollama / MCP / sandbox`, `knowledge → Ollama embed`.

## System context

```mermaid
flowchart LR
    U[Student / Moderator] -->|slash command| D[Discord]
    A[Admin · web] -->|HTTP| APP

    subgraph rust [apps/engine · apps/discord]
        BOT[discord]
        APP[engine<br/>agent · clients · routes]
    end
    D --> BOT
    BOT -->|HTTP / SSE| APP

    APP -->|OpenAI-compatible| OLL[Ollama · Qwen3]
    APP -->|HTTP| KN
    APP -.->|Phase 4| MCP[MCP servers]
    APP -.->|Phase 7| BW[apps/sandbox]

    subgraph knowledge [apps/knowledge — one image, two processes]
        KN[knowledge-api<br/>search · memory · conversations · sources]
        ING[knowledge-scraper]
    end
    KN --> PG[(PostgreSQL)]
    KN --> RD[(Redis)]
    KN --> QD[(Qdrant)]
    KN --> S3[(Object storage)]
    KN -->|OpenAI-compatible| EMB[Ollama · embed]
    ING --> WEB[Public ASU sites]
    ING --> PG
    ING --> QD
    ING --> S3
    ING --> EMB

    APP --> SEN[Sentry]
    APP --> AX[Axiom / OTLP]
```

Solid arrows exist or are in the current phase; dashed are later phases. Only the scraper touches the web; only `apps/knowledge` touches the datastores.

## Inside `engine`

`agent::harness` imports nothing else in the crate. `agent::model`, `agent::tools`, and `clients` import only `agent::harness`, never each other. `routes` and `wiring` compose them. This is a convention checked in review. Between apps it is enforced: `scripts/check-deps.sh` fails if `discord` and `engine` ever depend on each other, and `[workspace.lints]` applies the same code rules to both.

## Inside `knowledge`

`store/` is the only place a connection is opened. `index/` is shared by `knowledge-scraper` and `knowledge-api` because the chunker and embedding model used to *write* must be the ones used to *read*. `api/schemas.py` and `migrations/` are the wire and schema contracts with `engine`.

## Types

```rust
pub struct UserEvent {
    pub request_id: RequestId,
    pub source: EventSource,        // Discord, Http
    pub tenant_id: TenantId,        // the Discord guild; keeps data scoped
    pub user_id: UserId,
    pub channel_id: ChannelId,
    pub content: MessageContent,
    pub received_at: DateTime<Utc>,
}

pub struct RequestContext {
    pub request_id: RequestId,
    pub tenant_id: TenantId,
    pub user: UserIdentity,
    pub roles: Vec<Role>,
    pub conversation_id: ConversationId,
    pub deadline: Instant,
    pub cancel: CancellationToken,
    pub trace: TraceContext,
}

pub struct Evidence {
    pub source_id: SourceId,
    pub document_id: DocumentId,
    pub title: String,
    pub content: String,
    pub url: Option<Url>,
    pub fetched_at: DateTime<Utc>,
    pub score: f32,
}
```

Citations are built from `Evidence`, not parsed out of generated text.

## Traits

All in `engine/src/agent/harness`. Inputs and outputs are owned Sparky types; provider JSON stays inside adapters. `ModelProvider` and `Tool` are thin wrappers over Rig's `CompletionModel` and `Tool`, adding `RequestContext` and `RiskClass`.

```rust
#[async_trait]
pub trait ModelProvider {
    async fn generate(&self, ctx: &RequestContext, req: ModelRequest) -> Result<ModelStream, ModelError>;
}

#[async_trait]
pub trait Tool {
    fn definition(&self) -> ToolDefinition;   // name, description, JSON schema, RiskClass
    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError>;
}

#[async_trait]
pub trait Retriever {
    async fn retrieve(&self, ctx: &RequestContext, q: RetrievalQuery) -> Result<Vec<Evidence>, RetrievalError>;
}

#[async_trait]
pub trait ConversationStore {
    async fn load(&self, ctx: &RequestContext) -> Result<Vec<Message>, StoreError>;
    async fn append(&self, ctx: &RequestContext, turn: Turn) -> Result<(), StoreError>;
}

#[async_trait]
pub trait MemoryStore {
    async fn recall(&self, ctx: &RequestContext, q: MemoryQuery) -> Result<Vec<Memory>, MemoryError>;
    async fn write(&self, ctx: &RequestContext, m: MemoryCandidate) -> Result<Option<MemoryId>, MemoryError>;
    async fn forget(&self, ctx: &RequestContext, id: MemoryId) -> Result<(), MemoryError>;
}

#[async_trait]
pub trait Policy {
    async fn authorize(&self, ctx: &RequestContext, a: ProposedAction) -> Result<Decision, PolicyError>;
    // Decision: Allow | Deny(reason) | Confirm(ConfirmationRequest)
}

pub trait TraceSink {
    fn emit(&self, ctx: &RequestContext, e: TraceEvent);
}

#[async_trait]
pub trait Sandbox {   // Phase 7
    async fn run(&self, ctx: &RequestContext, t: BrowserTask) -> Result<BrowserResult, SandboxError>;
}
```

## Request lifecycle

1. `discord` receives the slash command and POSTs it to `engine` with the Discord identity and roles. Web and admin clients hit the same endpoint.
2. `engine` normalizes it into `UserEvent`, resolves roles and permissions → `RequestContext`.
3. Call `knowledge`: load conversation, recall memory, retrieve evidence if the query needs it.
4. Assemble context within a token budget, in fixed order: system instructions (versioned) → role/permissions → current request → relevant turns → memory → evidence → tool definitions. Tool output and evidence are truncated before old history is.
5. Call the model; validate structured output. One correction attempt on invalid output, then stop.
6. For each proposed tool call: `Policy::authorize` → Allow / Deny / Confirm.
7. Execute allowed calls with timeout and cancellation. Parallel when independent.
8. Loop until final answer, error, cancel, deadline, or step limit.
9. Persist turn, memory candidates that pass the write policy, and trace.
10. Reply with citations.

## Tool risk classes

| Class | Examples | Behavior |
|---|---|---|
| `ReadPublic` | search indexed pages, library hours | run |
| `ReadAuthenticated` | read a page in the user's own browser session | run after session authorization |
| `PrepareWrite` | draft an announcement, fill a form without submitting | run |
| `ExternalWrite` | post, create ticket, book, submit | confirm immediately before |
| `Destructive` | delete, cancel | confirm immediately before |
| `Forbidden` | another user's session, bypassing policy | deny |

A confirmation is bound to one exact action payload, is single-use and short-lived, states what happens / where / with what data / whether reversible, and is recorded in the trace. If the payload changes, confirm again. External writes are never auto-retried without an idempotency key.

## Knowledge

Ingestion runs offline in `knowledge-scraper`: fetch → content hash (skip if unchanged) → raw snapshot to object storage → extract → chunk → embed → index (Qdrant chunks + Postgres `source_versions`). Each document records canonical source, fetch time, content hash, parser/chunker/embedding versions, and its previous version on change.

Retrieval is `engine → POST /search` on `knowledge-api`: dense top-k (Qdrant, filtered by tenant and category) + BM25 top-k (Postgres FTS) → RRF fusion → rerank → `Evidence`. If nothing is found or the source is stale, the answer says so.

## Memory

| Kind | Content |
|---|---|
| Working | current request; not persisted |
| Conversation | turns and tool results |
| Episodic | a useful event from a prior interaction |
| Semantic | a stable inferred fact |
| Profile | approved preferences, interests, goals |
| Task | state to continue a multi-step job |

A candidate is written only if it is useful later, stable, belongs to this user, permitted by its sensitivity class, not a duplicate, and has an expiry. Recall filters by `tenant_id` and `user_id` before ranking; the interface cannot express a cross-user query. Users can view and delete their memory (Phase 4). Conflicting memories keep provenance and timestamps; newer and higher-confidence wins at assembly time, nothing is silently rewritten.

## Storage

| Data | Store |
|---|---|
| users, roles, conversations, messages, memories, source metadata and versions, jobs, confirmations | PostgreSQL (source of truth) |
| chunk embeddings with `source_id`, `version`, `category`, `fetched_at` | Qdrant (rebuildable) |
| rate limits, queue, short-lived cache | Redis |
| raw snapshots, model artifacts | object storage |
| browser session secrets | encrypted, separate namespace |
| traces | JSONL locally; OpenTelemetry in deployment |

Durable job state lives in Postgres, so a Redis restart loses nothing. Schema is `apps/knowledge/migrations`.

## Background jobs

Discord handlers never block on long work. Ingestion, embedding, browser tasks, reminders, evals, and trace processing run as jobs with id, owner, type, status, input ref, attempts, deadline, cancel state, result ref, and error category.

## Failure behavior

| Failure | Response |
|---|---|
| model unavailable | retry within deadline, then clear error |
| invalid tool call from model | one correction, then stop |
| tool timeout | cancel, trace, report |
| retrieval empty | say so; do not guess |
| sources conflict | show both with dates |
| confirmation denied or expired | do nothing |
| write result unclear | do not retry; inspect final state |
| knowledge-api unavailable | reject stateful requests rather than run without identity or policy |
| trace sink unavailable | continue only with a bounded local fallback |

## Tracing

One trace per request covering every model call, retrieval, memory access, tool call, policy decision, confirmation, and error. Records ids, model/prompt/template versions, evidence and memory ids, validated tool args, timings, tokens, final status. Never passwords, cookies, auth codes, tokens, or raw sensitive form values. Any request can be replayed from its trace against a chosen model, prompt, tool set, and retrieval snapshot; that is the eval harness.

## Sandboxed browser (Phase 7)

Separate worker (`apps/sandbox`), never inside the engine process. One isolated browser context per user session; the user completes login and MFA themselves; SparkyAI never asks for or stores a password. Allowlisted domains, blocked or quarantined downloads, size-limited structured observations, redacted action logs, session expiry and cleanup. CAPTCHA, MFA failure, expired session, or an unexpected page stops the task. Authenticated page content is never indexed or memorized. Requires explicit authorization before work begins (see roadmap out-of-scope).

## Deployment

Three images: `sparkyai-rust` (`engine` and `discord`; entrypoint selects), `sparkyai-knowledge` (`knowledge-api` and `knowledge-scraper`; entrypoint selects), `sparkyai-sandbox` (Phase 7, compose profile `sandbox`). CD rebuilds only the images whose inputs changed. Datastores run beside them in Compose; Ollama runs from `deploy/inference`. Split further only on a measured need: independent scaling, failure isolation, hardware, or a security boundary. Details: `deploy/README.md`.

## Open decisions

Reranker backend (llama.cpp `--reranking` vs in-process cross-encoder) · chat model size and quantization · Ollama concurrency under load · queue implementation · memory retention periods · moderator access to user conversations and traces · MCP servers in-process vs child process vs remote · when Qdrant moves to Piramid · app server host.

Record each as a short note under `docs/decisions/` when made.
