# Architecture

SparkyAI is a Discord copilot for the AI Society at ASU. It answers questions from public ASU sources, keeps conversation history and a profile of each user, and gates moderator actions behind policy and confirmation.

This document describes the current shape of the system and the rules it keeps. Order of work is in [ROADMAP.md](ROADMAP.md). Decisions are recorded in the commits that settle them.

## Stack

| Layer | Choice | Where |
|---|---|---|
| Engine, Discord bot, console | tokio, axum, serenity, ratatui, serde, thiserror, figment | `apps/engine`, `apps/discord`, `apps/cli` |
| Model and embed clients | Rig (`rig-core`) OpenAI-compatible client | `apps/engine/src/runtime/model/rig_openai.rs` |
| MCP | `rmcp` | `apps/engine/src/runtime/tools/mcp.rs` |
| Scraper | psycopg, httpx, BeautifulSoup, boto3, typer | `apps/scraper` |
| Page rendering | Firecrawl, self-hosted | `deploy/compose.yml` profile `crawl` |
| Web search | SearXNG, self-hosted | `deploy/compose.yml` profile `search` |
| Web | Vite, React, TypeScript, shadcn | `apps/web` |
| Post-training | Unsloth QLoRA, TRL, TensorBoard, GGUF | `apps/training/posttrain` |
| Evals | golden cases against the engine, baseline gate | `apps/training/evals` |
| Chat model | Qwen3 GGUF on `llama-server` | compose service `chat`, profile `model` |
| Embeddings | Qwen3-Embedding-0.6B, 1024 dimensions, on `llama-server` | compose service `embed`, profile `model` |
| Database, vector store, job queue | PostgreSQL 17 with pgvector | schema in `apps/scraper/migrations` |
| Object storage | S3-compatible, MinIO locally | `apps/scraper` |
| Cache | Redis 7, live query answers and their leases | `deploy/compose.yml` |
| Observability | OpenTelemetry spans to Phoenix, product events as spans, JSONL traces and logs under `.sparky/` | profiles `phoenix`, `metrics` |
| Config | `sparky.toml`, then `SPARKY_*` env vars | `sparky.toml`, `.env.example` |
| Build, gate | `just` recipes, pre-commit hook, CI | `justfile`, `.githooks`, `.github/workflows` |
| Deploy | Docker Compose, prod pulls GHCR images | `deploy/` |

## Rules

- Open models only, served by `llama-server` behind an OpenAI-compatible HTTP API.
- Facts come from retrieval or from a live source query, never from model weights.
- The engine never fetches a page. The scraper fetches, on a schedule or for a live query job, and is the only writer of the retrieval index.
- The engine and the scraper are the only processes with database connections. They share the schema in `apps/scraper/migrations`, not code.
- Every request carries its own `RequestContext`. There is no global mutable state.
- Every replaceable dependency is a trait in `engine/src/core/traits` with a test double in `core/tests/support`.
- The harness owns the loop, policy, context assembly, memory, and tracing. Provider JSON never leaves `runtime/model`.
- Model output is never written back as retrieval evidence.
- An action at or above `policy.confirm_from` is held for the caller's approval immediately before it runs.
- Credentials, cookies, and authenticated page content never enter the retrieval index, memory, or traces.

## Layout

```
apps/
  engine/         Rust bin. The agent, its HTTP surface, and the store adapters.
    src/core/       config (services, http, harness by domain), telemetry, types, traits, tests
    src/runtime/    harness (loop, prompt, memory, safety, compaction, trace), model (Rig client, slot limit, server props), tools (search, mcp, sandbox)
    src/stores/     postgres adapters: conversation, confirmation, knowledge (retrieval, query jobs), memory (memories, profile graph)
    src/routes/     chat (JSON and SSE), confirm, conversation, profile, openai (/v1), health, rate limit
    src/wiring.rs   builds every dependency from config and serves
  discord/        Rust bin. serenity bot and HTTP client of the engine. Never links it.
    src/core/       config, telemetry, types, tests
    src/bot/        client, addressed messages, memory commands, approvals, the streamed turn
    src/engine/     HTTP client of the engine and SSE frame parsing
    src/render/     the turn card, reply text, buttons and their custom ids
    src/access/     roles, permissions, and where a turn is answered
    src/analytics/  product events, one exported span each
  cli/            Rust bin sparky. Developer console: runs just recipes and compose services and tails them.
    src/core/       config, types, tests
    src/app/        console state, key map, control, rendering
    src/units/      unit catalog, process runner, log buffers, health probes
  scraper/        Python. Scheduled ingestion and the worker for live query jobs.
    core/           settings, types, telemetry, tests
    ingest/         fetch, extract, chunk, embed, tree, pipeline
    sources/        scheduled sources: one module per source with an extractor, pages.py for static pages
    query/          live query registry, parameter checks, runner, indexing of live results
    query/sources/  live query sources, one module each
    jobs.py         the job queue: handlers, lanes, scheduling
    store/          postgres and object storage, the only place a connection opens
    migrations/     the schema
  training/       Python. Datasets from Phoenix llm spans, evals with a baseline gate, SFT to GGUF.
  web/            Vite and React frontend and admin UI
deploy/           compose (dev and prod), Dockerfiles, inference, monitoring, search
docs/             ROADMAP.md, this file
.sparky/          ignored local state: traces, logs, training data, reports
```

Each Python app directory is its own importable package (`apps/scraper` is `scraper`) with no `src/` layer. `pyproject.toml` maps the package to `.` and lists its subpackages, so a new subpackage is added there.

Every app has a `core/` holding config, telemetry, data types, interfaces, and tests. Domain modules import from it. It imports nothing from them.

Language is never a folder. ASU domain (library, events) is never a folder either. It is a module in a source registry and a row in `sources` or `query_sources`.

## System context

```mermaid
flowchart LR
    U["Student or moderator"] --> DC["Discord"]
    DEV["Developer"] --> CLI["sparky console"]
    DEV --> OAI["OpenAI-compatible client"]

    subgraph apps ["apps"]
        BOT["discord"]
        ENG["engine"]
        SCR["scraper serve"]
    end

    DC --> BOT
    BOT -->|"HTTP and SSE"| ENG
    OAI -->|"/v1/chat/completions"| ENG
    CLI -->|"just, docker compose"| apps

    ENG -->|"completions"| CHAT["llama-server chat"]
    ENG -->|"embeddings"| EMB["llama-server embed"]
    ENG <-->|"conversations, index reads, jobs"| PG[("PostgreSQL and pgvector")]
    ENG -.->|"optional"| MCP["MCP servers"]
    ENG -.->|"run_sandbox"| SBX["sandboxd<br/>sandbox containers"]

    SCR <-->|"job claims, index writes"| PG
    SCR -->|"embeddings"| EMB
    SCR -.->|"tree summaries"| CHAT
    SCR --> FC["Firecrawl"]
    SCR --> SX["SearXNG"]
    SCR --> SITES["ASU sites and APIs"]
    FC --> SITES
    SCR --> S3[("MinIO snapshots")]

    BOT -->|"spans, product events"| OBS["Phoenix"]
    ENG -->|"spans"| OBS
    SCR -->|"spans"| OBS
```

Processes talk only at these edges. The engine and the scraper meet only in PostgreSQL: the engine queues a job and reads its row, the scraper claims it and writes the result. Only the scraper reaches the web. MCP servers are optional and none is configured by default.

The engine serves `/chat` and `/chat/stream` for the bot and `/v1/chat/completions` for OpenAI-compatible clients. All three run the same loop. Every route except health and `/v1/models` requires the bearer token in `SPARKY_ENGINE__SERVICE_TOKEN` and applies the per-user `http.rate_limit_per_min`.

The console starts and stops the other units and tails their output. It does not start a process whose port something else already serves, and says so instead.

## Inside engine

```mermaid
flowchart TD
    ROUTES["routes and wiring<br/>compose everything, own main"]
    HARNESS["runtime::harness<br/>agent: run, step, inputs, execute, conclude, task<br/>agent/call: thinking, relay, draft, thought, retry, spans<br/>agent/prompt: assemble, capability<br/>memory: detect, profile<br/>safety: guardrail, policy, redact<br/>compact, tools, trace"]
    MODEL["runtime::model<br/>rig_openai, limit, props"]
    TOOLS["runtime::tools<br/>knowledge/search, one file per source<br/>mcp, sandbox"]
    STORES["stores<br/>postgres, conversation, confirmation<br/>knowledge: retrieval, query<br/>memory: memories, profile"]
    CORE["core<br/>config, telemetry, types, traits, tests"]

    ROUTES --> HARNESS
    ROUTES --> MODEL
    ROUTES --> TOOLS
    ROUTES --> STORES
    ROUTES --> CORE
    HARNESS --> CORE
    MODEL --> CORE
    TOOLS --> CORE
    STORES --> CORE
```

`core` imports nothing else in the crate. `runtime::harness`, `runtime::model`, `runtime::tools`, and `stores` import only `core`. `routes` and `wiring` compose them. `scripts/check-deps.sh` enforces that the three Rust apps never depend on each other.

Folders nest by domain, and the same domain names repeat across `core/types`, `core/traits`, `core/tests`, `runtime`, and `stores`: agent, conversation, http, knowledge, memory, model, safety, tools, trace. A domain with one file at a level keeps that file flat. Data lives in `core/types`, interfaces in `core/traits`, and stateful objects beside their implementations.

`runtime::model::limit` wraps the chat client in a semaphore of `agent.model_slots` permits, matching `llama-server --parallel`. A call waits up to `agent.model_queue_wait_secs` for a slot, then fails as busy.

## Inside scraper

`store/` is the only place the scraper opens a connection. `migrations/` is the schema contract with the engine. The scraper writes `sources`, `source_versions`, `chunks`, `query_sources`, and job results. The engine reads the index and writes conversations, confirmations, the profile graph, and `source_query` jobs. `ingest/embed.py` uses the same model and dimension the engine queries with, so changing the embedding model means re-embedding every chunk.

`sources/` holds scheduled sources: a URL, a category, an interval, and an optional extractor. A source with an extractor has its own module. `sources/pages.py` lists static pages with none, indexed from the markdown Firecrawl returns. Scheduled fetches to one host are spaced `scraper.host_gap_secs` apart. `query/sources/` holds live query sources: parameters with their choices, and either a URL builder with an extractor or an `answer` function that reads several endpoints.

## Types

```rust
pub struct RequestContext {
    pub request_id: Uuid,
    pub tenant_id: String,
    pub user_id: String,
    pub roles: Vec<String>,
    pub conversation_id: Uuid,
    /// Public when anyone but the caller can read the answer.
    pub visibility: Visibility,
    pub deadline: Instant,
    pub cancel: CancellationToken,
    /// Live progress goes here while a caller is watching; None for a plain request.
    pub progress: Option<UnboundedSender<Progress>>,
}

pub struct Evidence {
    pub source_id: Uuid,
    pub chunk_id: Uuid,
    pub key: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub fetched_at: DateTime<Utc>,
    pub score: f32,
}
```

Citations are built from the `Evidence` that fit the prompt and from the pages tools read (`Answer.sources`), not parsed out of generated text. A `Citation` carries both halves of a source: `title` is what a client shows, the page title the scraper stored or the site name from `LiveSource::label`, and `key` is what it was, so a caller that has to match a source matches on the key.

## Traits

All in `engine/src/core/traits`. Inputs and outputs are owned types from `core/types`. The main ones:

```rust
#[async_trait]
pub trait ModelProvider {
    async fn generate(&self, ctx: &RequestContext, req: ModelRequest) -> Result<ModelResponse, ModelError>;
    // The same completion, sending each reasoning and text piece to deltas as it arrives.
    async fn stream(&self, ctx: &RequestContext, req: ModelRequest, deltas: UnboundedSender<ModelDelta>) -> Result<ModelResponse, ModelError>;
}

#[async_trait]
pub trait Tool {
    fn definition(&self) -> ToolDefinition;   // name, description, JSON schema, risk, sequential, timeout
    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError>;
}

#[async_trait]
pub trait Retriever {
    async fn retrieve(&self, ctx: &RequestContext, query: &RetrievalQuery) -> Result<Vec<Evidence>, RetrievalError>;
}

#[async_trait]
pub trait Embedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError>;
    fn dim(&self) -> usize;
}

#[async_trait]
pub trait SourceQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError>;
    async fn run(&self, ctx: &RequestContext, request: &QueryRequest) -> Result<QueryOutcome, QueryError>;
}

#[async_trait]
pub trait ConversationStore {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError>;
    async fn owns(&self, ctx: &RequestContext) -> Result<bool, StoreError>;
    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError>;
    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError>;
    async fn latest(&self, ctx: &RequestContext, channel_id: &str) -> Result<Option<Uuid>, StoreError>;
    async fn end(&self, ctx: &RequestContext, channel_id: &str) -> Result<u64, StoreError>;
}

#[async_trait]
pub trait MemoryStore {
    async fn recall(&self, ctx: &RequestContext, q: &MemoryQuery) -> Result<Vec<Memory>, StoreError>;
    // No write method yet.
}

#[async_trait]
pub trait Policy {
    async fn authorize(&self, ctx: &RequestContext, action: &ProposedAction) -> Decision;
    // Decision: Allow | Deny { reason } | Confirm(ConfirmationRequest)
}

#[async_trait]
pub trait Guardrail {
    async fn check(&self, ctx: &RequestContext, stage: Stage, text: &str) -> Verdict;
}

pub trait TraceSink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent);
}
```

The rest follow the same pattern: `Compactor`, `ConfirmationStore`, `ProfileGraph`, `FactDetector`, `Router`, `QueryCache`, `Sandbox`.

## Request lifecycle

```mermaid
sequenceDiagram
    actor U as Student
    participant B as discord
    participant E as engine
    participant PG as PostgreSQL
    participant M as llama-server
    participant S as scraper

    U->>B: @mention, a reply in the thread, or a direct message
    B->>B: resolve roles, destination, visibility
    B->>E: POST /chat/stream with bearer token and traceparent
    E->>PG: ensure conversation, load recent turns
    opt history over its budget
        E->>M: chat agent compacts the oldest turns
    end
    E->>PG: recall memory and profile graph, private turns only
    E->>M: embed the question
    E->>PG: dense and lexical search
    loop one step per model call
        E->>M: streamed completion with tool schemas
        M-->>E: reasoning, text, tool calls
        E-->>B: SSE progress, thinking line, answer draft
        opt search tool call
            E->>PG: queue source_query job
            S->>PG: claim, fetch, write result
            PG-->>E: result text
        end
    end
    E->>PG: append the new turns
    E-->>B: SSE answer, then done
    B->>U: final card with answer, sources, memories
    E->>E: spawn profile writer, detached
```

1. `discord` receives a message addressed to it and posts it to the engine with the Discord identity, role names, the channel it answers in, the visibility of that answer, `continue_channel`, and the message of its own a reply answers. A question is never a slash command; the remaining commands only read and clear memory.
2. The engine checks the bearer token and the per-user rate limit, then builds a `RequestContext`. A given `conversation_id` must belong to the caller. Without one, `continue_channel` continues the caller's newest open conversation in that channel at that visibility, and otherwise a new conversation starts.
3. The loop loads inputs once: the last `agent.history_turns` turns, compacted if they overflow the history budget; memories and the profile graph, unless the request is public and `agent.recall_in_public` is off; and evidence from retrieval for the question, unless the router skipped it. None of the three is the turn: a memory or retrieval read that fails is logged and the turn runs without it, retrieval taking `Skipped::Unavailable` so the prompt says so and the search tools carry the answer.
4. The loop runs steps until it stops. On `/chat/stream` each progress event goes out as an SSE frame while the step runs. A turn about to answer after a tool failed, with `run_sandbox` registered and untried, is handed back once with `prompt.sandbox_retry_line` instead: a failed tool is not an answer, and the student hears that nothing was found only once the last route has been taken.
5. Every exit appends what was said, hands the user's text to the profile writer, and builds the answer with its citations, tool runs, memories (empty for a public answer), usage, and cost. A tool call and its result answer the question being asked now, so they run the loop and are not stored: carrying a fetched page into later turns spends the history budget on it and offers the model a stale copy of a page it can fetch again.
6. An answer with status `AwaitingConfirmation` carries a token. `POST /confirm` with that token runs the held action and resumes the loop from its result.

## Prompt assembly

Each step assembles the prompt in a fixed order:

1. System instructions, the role line, and today's date.
2. The capabilities section, if it fits `agent.capabilities_budget_tokens`.
3. Memory, within `agent.memory_budget_tokens`.
4. Evidence, numbered for citation, within `agent.evidence_budget_tokens`. Each entry is labelled a stored copy with its fetch date and age, and `prompt.evidence_header` tells the model to call the matching search tool when the question needs current information or the entries do not answer it. With no evidence, one line takes its place: `prompt.no_evidence_line` when retrieval ran and found nothing, `prompt.no_retrieval_line` when the router called the turn small talk, and `prompt.live_only_line` when it called the answer current.
5. History, newest first within `agent.history_budget_tokens`, never starting on an orphaned tool result.
6. The user's input, then the model and tool turns of this request.

Tool schemas are sent beside the messages and are charged to `agent.prompt_budget_tokens` first. Evidence and history are capped by what remains. The system prompt, the input, and this request's tool exchange are never dropped. When this request's tool results outgrow the room left after the fixed sections and the input, each is cut to a share of it, smallest first, and ends in `prompt.result_cut_line`. Token counts are estimates from `agent.chars_per_token`. MCP schemas are compacted and, by default, show only required properties.

## Agent loop

```mermaid
flowchart TD
    TOP(["next step"]) --> LIMIT{"cancelled, past deadline,<br/>or max_steps reached?"}
    LIMIT -->|yes| STOP(["Cancelled, Deadline, or StepLimit"])
    LIMIT -->|no| ASM["assemble prompt"]
    ASM --> THINK{"thinking decision"}
    THINK --> CALL["model call<br/>no tools when an answer is forced"]
    CALL --> RELAY["relay while streaming<br/>reasoning line, answer draft blocks"]
    RELAY --> SPENT{"thought, but no answer<br/>and no tool call?"}
    SPENT -->|yes| PLAIN["same call without thinking"]
    SPENT -->|no| GUARD
    PLAIN --> GUARD{"guardrail passes?"}
    GUARD -->|no| BLOCKED(["Blocked, replacement text"])
    GUARD -->|yes| CALLS{"tool calls?"}
    CALLS -->|no| ANSWER(["Answered"])
    CALLS -->|yes| POLICY{"Policy::authorize<br/>each call"}
    POLICY -->|Confirm| HOLD(["AwaitingConfirmation"])
    POLICY -->|"Deny or unknown tool"| FEED
    POLICY -->|Allow| REPEAT{"same name and arguments<br/>already run?"}
    REPEAT -->|"some calls are new"| EXEC["run new calls with timeouts<br/>parallel unless one is sequential"]
    REPEAT -->|"all repeats, first time"| FORCE["force an answer on the next step"]
    REPEAT -->|"all repeats, already forced"| STALLED(["Stalled"])
    EXEC --> FEED["append tool results as messages"]
    FEED --> TOP
    FORCE --> TOP
```

The loop owns every stopping condition. It checks cancellation, the deadline, and `agent.max_steps` before each step.

Thinking is decided per model call by `agent.thinking`. `on` and `off` apply to every call. In `auto` the first matching rule wins: a call with no tools does not think; a step after tool results thinks when `after_tools` is set; a question with a cue word, longer than `max_quick_chars`, or with no evidence thinks; anything else does not. A call that thinks gets `model.max_tokens`, and one that does not gets `model.max_tokens_without_thinking`. A call that thought and returned neither an answer nor a tool call is made once more without thinking when `retry_without` is set. The `llm` span records `sparky.thinking` and `sparky.thinking_reason`. Prompted sub-agents never think.

With `agent.stream` on, the model call streams. Reasoning updates the thinking line as it grows. When answer text begins, the thinking line becomes the full thought, clipped to `agent.progress_thought_chars`. Answer text is released as a draft at the end of a sentence or line, or at a word break past `agent.stream_block_chars`, and every block passes the guardrail before it is sent. A draft is withdrawn when the call ends in tool calls, fails, or a block is refused. A step that answers without showing a thought takes its thinking line back.

Every response passes the guardrail: the answer stage for a final answer, the capability stage for a response with tool calls and text. `Policy` decides whether a typed action may run by its risk class. The guardrail decides whether a response may proceed at all.

Tool calls are authorized before any runs. A denial or an unknown tool name becomes a tool result the model reads. The first call that needs confirmation is stored in `confirmations` and ends the run. Allowed calls that repeat an earlier call with identical arguments are not run again, and the model is told so. When every call in a step is a repeat, the next step offers no tools and adds `prompt.answer_only_line`. If that step repeats again, or answers with nothing, the run ends as `Stalled`.

Allowed calls run in parallel, each under its own timeout (the tool's declared timeout or `agent.tool_timeout_secs`, capped by the request deadline). If any call is to a tool marked sequential, the step runs its calls in order. A tool error, including `InvalidArguments`, is fed back as the tool result so the model can correct the call on its next step.

A retryable model error (transport, 5xx, 429) is retried with backoff up to `agent.max_model_retries`. A model timeout ends the run as `Deadline`, cancellation as `Cancelled`, and any other error keeps the turns and returns the error.

## Prompted sub-agents

The harness runs more than one prompt. Each sub-agent is a `runtime::harness::agent::task::Task`: its own instructions, one model call, no tools, no thinking, and a typed result. The loop does not use `Task`, because it has tools and stopping conditions.

| Agent | When | Reads | Writes |
|---|---|---|---|
| Sparky (the loop) | every request | evidence, memory, history, capabilities | the turn |
| Chat | loaded history overflows its budget | the turns being replaced | one compacted turn |
| Graph | after a turn, when the detector passes it | the user's text | profile facts |
| Reconcile | a new fact collides with recorded ones | the new fact and the candidates | which recorded facts to withdraw |

## Capabilities

`runtime::harness::agent::prompt::capability` renders one `<capabilities>` list in the prompt. Each entry names its kind.

| Kind | Executed by | Risk |
|---|---|---|
| `tool` | a built-in `Tool`: `search_knowledge`, `search_live`, `run_sandbox` | declared per tool |
| `mcp` | a remote MCP server | derived from the tool name |

`tools.disabled` removes tools by name at registration. The engine refuses to boot when the capabilities section exceeds its budget or the tool schemas take more than half of `agent.prompt_budget_tokens`. It also refuses when the prompt budget plus `agent.prompt_estimate_headroom` and `model.max_tokens_without_thinking` do not fit one `llama-server` slot.

`run_sandbox` runs a command in a container with a read-only root, memory, CPU, and process limits, a non-root user, and a workspace of `sandbox.workspace_mb` mounted `noexec` at /tmp. The image is `deploy/docker/sandbox.Dockerfile`: python3 with requests, bs4, lxml and pandas, curl, wget, jq, ripgrep, sqlite3, pdftotext and the usual text tools.

With `sandbox.egress` off the container runs with `--network none`. With it on, it joins `sandbox.egress_network`, which the engine creates `--internal` at boot and refuses to use if it exists and is not internal, and it is handed `HTTP_PROXY` and `HTTPS_PROXY` pointing at `sandbox.egress_proxy_name`. That proxy (`deploy/docker/sandbox-proxy.Dockerfile`, `deploy/sandbox/squid.conf`) is the only container on both that network and the outside: it allows ports 80 and 443 and refuses private, loopback, link-local and reserved destinations, so a command reaches public sites and never the datastores, the host, or cloud metadata. HTTPS passes through as a tunnel, so the proxy sees the host and not the request. What the sandbox reads is never indexed. A call that names a session reuses a container, so files in the workspace persist between calls; the container is removed once it has been idle for `sandbox.session_idle_secs`, and a caller holding `sandbox.max_sessions` loses their least recently used one. Session containers are named from a hash of the tenant and user, so one caller cannot reach another's session.

The engine needs a container runtime. Under compose that is the `sandboxd` service, a daemon of its own reached over `DOCKER_HOST`, so driving it is not driving the host runtime; on a developer host it is the local `docker`. The runtime is probed once at boot: `sandbox.required` decides whether an unreachable one fails boot or leaves `run_sandbox` unregistered, so the model is never offered a tool that always fails.

A tool result longer than `agent.tool_result_to_file_chars` is written to the workspace and replaced by its head plus the path, so a long result stops riding in the conversation for every later step of the turn. It is off at `0`; the full result still reaches the trace either way.

## Compaction

Without compaction, history is trimmed to its budget by dropping the oldest turns. With `compaction.enabled`, compaction runs only when the loaded history, a stored summary included, is over `agent.history_budget_tokens`. It keeps the newest turns within `compaction.keep_share` of that budget, moved forward to start on a user turn so no exchange is split, and the Chat agent replaces everything older, the previous summary included, with one compacted turn that is stored, so the next request starts from it. Boot rejects a `keep_share` whose kept turns plus `compaction.max_tokens` exceed the history budget, so the summary and the kept turns always fit together, and the turns that follow have the rest before the next compaction. Assembly places a leading summary before trimming the turns after it. A failed compaction falls back to trimming.

A compacted turn is model output. It is stored with role `summary`, so a replayed conversation can tell it apart from what the user and the assistant said. It is never retrieval evidence, and the turns it replaced stay in `messages`.

A summary row records `covers_seq`, the `seq` of the last message it replaces. Loading history reads the newest summary and then up to `agent.history_turns` messages whose `seq` is greater than its `covers_seq`. The turns a compaction kept are stored before the summary, so they load again on the next request. A summary without `covers_seq` covers every message before its own `seq`.

The system prompt is not a stored message. It is rebuilt on every request and always comes first, so compaction never replaces it.

## Profile graph

Flat memory rows record what a user said. The profile graph records entities, the relations between them, and an embedding per node, so recall can start from a relation.

Extraction never runs in the request path. When a run ends, the loop spawns the profile writer with the user's text and returns. The writer has its own context and a deadline of `profile.timeout_secs`.

The gate is rules, not a model call, because it runs on every turn. `RuleDetector` passes a sentence with a first-person marker next to a stative cue and rejects questions and turns shorter than `profile.detector.min_words`. Only a turn that passes reaches the Graph agent. Facts below `profile.min_confidence` or with an empty label are dropped. A trained classifier can replace the rules through the `FactDetector` trait.

With `profile.reconcile` on, a new fact is checked against facts already recorded for the same subject and relation. The Reconcile agent names the recorded facts the new one makes false, and those are withdrawn. Two facts that can both be true are both kept. An answer that cannot be parsed, or a failed call, withdraws nothing.

Recall filters by `tenant_id` and `user_id` before ranking. Users list and delete what the graph holds with `/memory` and `/forget` (`POST /profile/list`, `POST /profile/forget`). Relations cascade from the node they run through.

## Memory

| Kind | Content | Status |
|---|---|---|
| Working | the current request | not persisted |
| Conversation | what was said, the question and the answer | `messages` |
| Episodic, semantic, profile, task | durable facts about a user | `memories`, read by recall |
| Profile graph | entities and relations | `profile_nodes`, `profile_edges` |

The engine recalls `memories` rows, unexpired and ordered newest and most confident first, and appends profile graph nodes and relations. It does not write `memories` rows yet. The schema carries `sensitivity`, `confidence`, `source_msg`, and `expires_at` for when it does.

A public request recalls no memory and no profile graph unless `agent.recall_in_public` is set. Only an answer no one else can read is private. A public answer never lists the memories its prompt carried.

## Tool risk classes

| Class | Examples | Default behavior |
|---|---|---|
| `ReadPublic` | `search_knowledge`, `search_live` | run |
| `ReadAuthenticated` | a page inside the user's own session | deny unless `policy.allow_authenticated_reads` |
| `PrepareWrite` | `run_sandbox`, a draft, a form filled without submitting | run |
| `ExternalWrite` | post, create a ticket, book, submit | require a `policy.write_roles` role, then confirm |
| `Destructive` | delete, cancel | require a `policy.write_roles` role, then confirm |
| `Forbidden` | another user's session, bypassing policy | deny |

The classes are ordered as listed. `policy.write_roles` gates `ExternalWrite` and above. `policy.confirm_from` names the lowest class held for approval, so a deployment can hold drafts too while a new tool is being trusted. `Forbidden` is denied regardless of settings. Defaults are `["MANAGE_GUILD"]`, `external_write`, and authenticated reads off.

A confirmation is bound to a hash of the exact arguments, belongs to the caller who was asked, is single use, and expires after `agent.confirmation_ttl_secs`. Its summary names the tool, the arguments, and whether the action can be undone. The policy decision is recorded in the trace.

## Knowledge

The retrieval index is `chunks`, written by the scraper and read by the engine. Two kinds of job write it: a scheduled run of a registered source, and the indexing of a page a live query fetched. Both go through the same pipeline in `ingest/pipeline.py`.

```mermaid
flowchart TD
    RUN["source_run job<br/>a registered source fell due"] --> FETCH["fetch<br/>Firecrawl, or httpx when scraper.fetcher is http"]
    LIVE["live_index job<br/>text a live query already fetched"] --> PAGE["pick the source row<br/>scheduled source with that URL,<br/>or query key plus URL digest"]
    FETCH --> HASH{"content hash equals<br/>the latest version?"}
    PAGE --> HASH
    HASH -->|yes| SKIP(["unchanged, nothing written"])
    HASH -->|no| SNAP["raw snapshot to object storage"]
    SNAP --> EXTRACT["extract text<br/>source extractor or shared heuristic"]
    EXTRACT --> CHUNK["chunk<br/>scraper.chunk_chars"]
    CHUNK --> FLOOR{"text above the quality floor<br/>of the last version?"}
    FLOOR -->|no| REFUSE(["PipelineError, old index kept"])
    FLOOR -->|yes| EMBED["prefix the page title<br/>embed the batch, 1024 dims"]
    EMBED --> WRITE["insert source_versions row<br/>replace the source's chunks"]
    WRITE --> TREE{"scraper.tree_enabled?"}
    TREE -->|no| DONE(["committed"])
    TREE -->|yes| SUM["cluster, summarize, embed<br/>summary rows by level"]
    SUM --> DONE
```

Each version records content hash, snapshot key, parser, chunker and embedding versions, text length, chunk count, and the previous version. The quality floor refuses a run whose text is under `scraper.quality_floor_ratio` of the last version, once that version had at least `scraper.quality_floor_min_chars`. An extractor break that returns only navigation is caught this way. `scraper run <source> --force` accepts such a run.

### The gate in front of retrieval

`retrieval.router` decides whether a question is retrieved for at all, through the `Router` trait in `core/traits/knowledge/route.rs`. `RuleRouter` is rules, not a model call, because it runs on every turn and a model call before retrieval would cost more latency than the retrieval it skips. A trained classifier can replace the rules through the trait, as `FactDetector` allows for the memory gate.

A turn holding a cue from `retrieval.router.live` skips retrieval as `Skipped::Live`: the answer has to be current, so a search tool answers it and a stored copy would be out of date. A turn of at most `retrieval.router.max_chitchat_words` that holds a marker from `retrieval.router.chitchat` and no live cue skips it as `Skipped::Chitchat`: it asks for no ASU fact. Live wins over chitchat. Anything else is retrieved for. Both lists match whole words, lowercase, punctuation ignored, and an empty list uses the built-in one.

A skip emits `TraceEvent::RetrievalSkipped` and changes two things downstream: the prompt gets the line for that reason in place of `prompt.no_evidence_line`, and the call does not think, because the thinking rule that reasons over missing evidence applies only when retrieval actually ran.

Retrieval runs in the engine before the first model call, over the same rows.

```mermaid
flowchart TD
    Q["question"] --> EMB["embed"]
    EMB --> DIM{"dimension matches<br/>the index?"}
    DIM -->|no| ERR(["RetrievalError"])
    DIM -->|yes| DENSE["dense leg<br/>cosine distance, HNSW<br/>drop past retrieval.max_distance"]
    DIM -->|yes| LEX["lexical leg<br/>websearch_to_tsquery, ts_rank_cd, GIN"]
    DENSE --> RRF["reciprocal rank fusion<br/>retrieval.rrf_k"]
    LEX --> RRF
    RRF --> MIN["drop below retrieval.min_score"]
    MIN --> TREE["drop a row whose summary<br/>ranked higher"]
    TREE --> TOPK["top retrieval.top_k"]
    TOPK --> WIN["widen each hit by<br/>retrieval.window rows"]
    WIN --> EV["Evidence"]
```

Each leg pulls `retrieval.candidates` rows from the caller's tenant and the `public` tenant. The dense leg drops a row farther than `retrieval.max_distance` from the question, so a question the index does not cover gets no evidence. Fusion combines ranks, not the legs' incompatible raw scores. Either leg can be turned off, but not both. A reranker is deferred until evals justify it.

### Sentence windows

What is matched and what is read are different sizes. Chunks are cut at `scraper.chunk_chars`, narrow enough that a match points at the passage that answers rather than at a whole section, and `scraper.chunk_overlap_chars` is 0 because the window supplies the continuity overlap used to.

After the top `retrieval.top_k` rows are chosen, each one is read back with the `retrieval.window` rows either side of it, joined in ordinal order into one passage. Windows that overlap or touch merge, so two hits a row apart cost one passage rather than two copies of the middle. A span is handed back once however many of its rows were hit, and two hits far apart on one page stay two passages. Summaries are not widened: a summary already covers its chunks. `retrieval.window = 0` hands each hit back alone.

Chunk ordinals are contiguous within a version, and tree summaries are ordinalled after the leaves, so widening is a range read on `(version_id, ordinal)` and reads only `level = 0` rows. Changing `scraper.chunk_chars` or `scraper.chunk_overlap_chars` changes `chunker_version`, which reindexes a source on its next run. Citations are unaffected: a `Citation` is the source title and URL, not an offset.

### Hierarchical index

With `scraper.tree_enabled`, ingestion clusters the chunks of one source, writes a model summary of each cluster as a new row, embeds it, and repeats up to `scraper.tree_max_level`. A summary row lives in `chunks` beside the leaves, with a `level` above 0, and each leaf points at its summary through `parent_id`. The summaries use the `[summary]` model on the chat server.

Retrieval searches every level at once, so a query lands on whatever granularity answers it (the RAPTOR collapsed-tree approach). After fusion, a row whose summary already ranked higher is dropped. A chunk that outranks its own summary keeps both.

The tree is off by default. It costs a model call per cluster per level.

## Live source queries

Scheduled ingestion keeps the index current on an interval. A live query answers from a source now, with parameters the model picks: a term, subject and level of the class catalog, a scholarship search filtered by the student's situation, study room slots on a date, the next shuttle at each stop, a building on the campus map, a web search, or a refetch of this week's library hours.

The engine offers two search tools, not one per source: `search_knowledge(query, source?)` reads the stored index and `search_live(query, source?)` queues a fetch. `source` is a filter, never a forced guess: leaving it out searches every source, and each variant carries a few words from `LiveSource::hint` so the model can pick one when it knows. `search_knowledge` offers only the sources the scraper indexes, and maps the key it is given to the `chunks.category` it filters on. `search_live` answers a call that names no source from `tools.live_default_source`, which is `web`.

The query string is the whole request. With one tool per source the tool name carried half the intent; with two tools it does not, so `tools.query_description` is the highest-leverage wording in the prompt and says in a bad-then-good pair what a self-contained keyword query looks like. `search::params_for` turns that one string into the parameters the scraper takes: the query fills the source's text parameter, a choice the query names fills a choice parameter, a `YYYY-MM-DD` in the query fills a date, a required date falls back to today at `prompt.utc_offset_hours`, and `Courses::derived` reads the term out of the query or takes the one running today. A required parameter nothing filled comes back to the model naming what the query has to say. `LiveSource::narrow` cuts what a source's site cannot match out of the query before it fills that parameter: `Courses` sends the class search the course code alone, because its keyword box matches subject, catalog number and title and nothing else. Two schemas cost about 790 estimated tokens where one per source would cost over 1700.

`LiveSource::freshness` declares whether a source's answers are stored, the scraper publishes the same fact as `query_sources.indexed`, and `search::conforms` names a disagreement at boot the way it does for parameters. The prompt tells the model to use the evidence first, to call `search_knowledge` when the evidence does not answer, and to call `search_live` whenever the question is one only it can answer, because nothing in the prompt ever holds that.

```mermaid
sequenceDiagram
    participant M as model
    participant T as engine search tool
    participant PG as PostgreSQL jobs
    participant L as scraper live lane
    participant BG as scraper background lane
    participant W as source site or SearXNG

    M->>T: search_live with a query and a source
    T->>T: turn the query into the parameters the source takes
    T->>PG: insert source_query, priority 0, deadline
    T->>PG: pg_notify source_query
    PG-->>L: notification wakes the lane
    L->>PG: claim with for update skip locked
    L->>W: fetch the page or read the endpoints
    L->>PG: mark done with result, queue live_index at -10, one commit
    T->>PG: poll status every query.poll_ms
    PG-->>T: done, result text
    T-->>M: live result text and citation
    BG->>PG: claim live_index
    BG->>BG: index the page through the pipeline
    BG->>PG: write chunks
    Note over T,PG: unclaimed after query.claim_secs, the job is cancelled and the tool reports the scraper is not running
```

A source has two halves, one file each. The engine side, `runtime/tools/knowledge/search/<source>.rs`, declares the hint, the label a citation of it carries, the chunks category, the parameters, what each accepts (text, one of fixed choices, any of fixed choices, a date, a flag), and any extra check. The query is turned into those parameters and checked there, so a bad call is corrected by the model without queueing a job. The two tools themselves live in `live.rs` and `stored.rs`. The scraper side, `apps/scraper/query/sources/<source>.py`, turns those parameters into a fetch: one URL and an extractor, or an `answer` function that reads several endpoints itself (the shuttle tracker, campus map layers, news and video feeds, SearXNG).

`query_sources` is the registry the scraper publishes when `scraper serve` starts: each key with its parameters and choices. At boot the engine checks every source in its catalog and logs a warning when a source and the published one disagree on a parameter name, whether it is required, whether it takes a list, or a choice. A source the scraper has not published is still offered. The scraper checks the parameters of every query again before it runs it.

Any scraper failure (unknown source, missing parameter, unreadable page, an exception) marks the job failed. The tool returns the reason as `InvalidArguments`, which the loop feeds to the model. A caller that is cancelled or runs out of time cancels its job. The text handed back is held to `scraper.query_max_chars`, and the page is cited in `Answer.sources`.

The `web` source uses the same path against the open web. The scraper queries a self-hosted SearXNG (`just search`, `[search]` in `sparky.toml`) over Google, Brave, and Bing, and returns titles, links, dates, and snippets, cited as the Google search for the query. DuckDuckGo is excluded because it answers self-hosted searches with a CAPTCHA.

Not covered: X and Instagram posts (the public X timeline endpoint serves old posts and Instagram requires a login) and anything behind a student login, such as Workday jobs.

### Caching live results

`query.cache` puts a Redis cache in front of every live query, as `CachedQueries` wrapping `SourceQueries`. It does two separate things.

**Reuse.** An answer within its lifetime is returned without a `jobs` row, without waking the scraper, and without a fetch. The key is a UUIDv5 over the tenant, the source key, and the parameters sorted by name, so the same question from different students is the same key, and the order the model happened to write the arguments in does not matter. A reused answer carries the time it was fetched, and `LiveSearch` renders that age into the tool output, so the model can never present a stored answer as current.

**One fetch per query.** Whatever the lifetime, the first request to ask for a query takes a lease (`SET NX`) and fetches; every request that arrives while it runs waits on that lease and reads the answer it writes. A hundred students asking about CSE 310 at once is one fetch of the ASU catalog, not a hundred. Coalescing costs no freshness at all, because a request that waited gets an answer fetched after it asked. Reuse does cost freshness, which is why every source the scraper never indexes defaults to a lifetime of zero.

`query.cache.handoff_secs` is the floor under every lifetime: an answer has to outlive its own fetch for the requests that waited on it to read it, so a source at zero is reused for that long and no longer. `lease_secs` must cover `query.timeout_secs` or a lease can expire mid-fetch and let a second request fetch the same query; `Config::validate` rejects that at boot, along with a cache enabled without a `redis` section.

A refusal is cached for the handoff window too, so a source rejecting an argument is not hammered. Every other failure releases the lease, because a timeout or an absent scraper says nothing about the query and the next request should start over.

The cache is never load-bearing. Any Redis error is logged and the request fetches as if the cache were not there, recorded on the trace as `CacheOutcome::Unavailable`. Caching is safe on this path by construction: a source query only ever reads, the search tools are all `ReadPublic`, and anything that writes goes through `Policy`, never through `SourceQueries`.

### Keeping the database standing under a spike

Every live query costs the one table every process writes to. Four things bound that cost, and only one of them is Redis.

**The wait backs off.** The engine waits on the scraper by polling the `jobs` row. `query.poll_ms` is the first wait and it doubles up to `query.poll_max_ms`, so a 30 second fetch costs under 40 round trips rather than one every 100ms for its whole duration. This is the largest single reduction and it needs nothing shared.

**The number of fetches is capped.** `query.max_in_flight` caps how many live queries reach the database at once, across every replica, as a sorted set in Redis scored by when each slot was taken. Over the cap a search tool is refused with `QueryError::Busy`, which reaches the model as a tool error naming the source. Refusing beats queueing onto the connection pool until `postgres.acquire_timeout_secs` expires: the model can answer without that source, and the student gets an answer rather than a timeout. Keep the cap under `postgres.max_connections`. A slot whose holder never gives it back falls out of the set once it is older than the lease, so a killed replica does not leak capacity.

`AdmittedQueries` sits **under** `CachedQueries`, which is what makes a reused answer free: a cache hit and a request that waited on another request's lease take no slot and touch no database. Only an actual fetch is counted.

**Background indexing sheds load.** Past `scraper.index_backlog_limit` queued `live_index` jobs, a live result is answered but not indexed, and the drop is logged. Answering is what a student is waiting on, and the source is refetched on its schedule anyway. The check reads no further than the limit, so measuring the backlog is never itself the expensive part.

**Finished jobs are removed.** `scraper.job_retention_hours` bounds the table: each scheduling cycle deletes at most `scraper.job_prune_batch` terminal jobs older than the retention, so claiming stays fast and one cycle never takes an unbounded number of row locks. Queued and running jobs are never pruned, however old.

Two partial indexes serve this, added in `0013_job_queue_pressure.sql`: `(kind, priority desc, created_at) where status = 'queued'` for claiming and for the backlog check, and `(updated_at) where status in ('done','failed','cancelled')` for pruning. The first carries `kind`, which the old queued index did not, so it replaces it. Both are built `CONCURRENTLY`, which Postgres refuses inside a transaction, so the migration runner sends a file marked `-- concurrent:` one statement at a time outside one.

### Indexing live results

A live result answers its caller first and then becomes evidence. The job that stores the answer also queues a `live_index` job with the full fetched text, in the same commit. The background lane runs it through `pipeline.index_page`: hash, snapshot, extract, chunk, quality floor, embed, write, tree. A page whose URL is a scheduled source refreshes that source. Any other page gets its own `sources` row keyed by the query source and a digest of the URL, so the same search refreshes the same rows. The scheduler runs only registered sources and never refetches such a page.

A query source sets `index = False` when its answer goes stale within minutes or is not ASU content: `shuttles`, `study_rooms`, and `web`. Those are the sources `search_knowledge` never offers as a filter, because nothing of theirs is ever in the index. `scraper.index_live_results` turns indexing off entirely. A failure to index fails only the `live_index` job and never reaches the caller.

## Job queue

The scraper does all its work from the `jobs` table in one process, `scraper serve` (`apps/scraper/jobs.py`).

| kind | queued by | priority | lane |
|---|---|---|---|
| `source_query` | the engine, for a `search_live` call | 0 | live |
| `live_index` | the scraper, with the answer to a `source_query` | -10 | background |
| `source_run` | the scraper, when a registered source falls due | -20 | background |

A claim takes the highest priority first, then the oldest, skipping jobs past their deadline, with `for update skip locked`, so several lanes and processes can claim at once. The live lane claims only `source_query` and wakes on the engine's `pg_notify`. The background lane claims the other kinds, so a long scrape never delays a live query. Both lanes also poll every `scraper.serve_poll_secs`.

Every `scraper.schedule_every_secs`, the main thread queues a `source_run` for each registered source that is due by its `fetch_every` and last attempt. A partial unique index allows at most one queued or running `source_run` per source. The same pass requeues background jobs left running longer than `scraper.job_lease_secs` by a stopped process.

`scraper status` shows each source and the queue by kind and status. `scraper run <source>` runs one source directly, outside the queue; `--category <name>` runs a category and `--all` every source.

## Discord surface

| Entry | Where the answer goes | Visibility |
|---|---|---|
| mention in a text channel | a thread opened from the message, or inline if the thread cannot be made | public |
| reply to a Sparky message in a thread | that thread | public |
| any other message in a thread | nowhere; the bot stays quiet | |
| mention or reply in a text channel | a thread opened from the message | public |
| direct message | the direct message channel | private |

A thread is the conversation. Inside one, only a reply to something Sparky said continues it, so
people talk in the thread without Sparky answering every line, and the message replied to rides the
request as `reply_to` and is quoted into the prompt under `prompt.reply_header`, capped by
`agent.reply_budget_tokens`. A reply to a person, or to another bot, is not a turn.

A direct message is private, which is what `agent.recall_in_public` keys on: personal memory and the
profile graph reach a direct message and never an answer in a server. `bot.direct_messages` turns
the direct channel off.

Up to `bot.max_images` image attachments ride the request and are attached to the user turn as
image blocks. The engine filters them again in `Attachment::accepted`, because the HTTP surface
takes any caller holding the bearer token. Only the link travels, so the model server fetches it,
and history stores none of them: the link the edge issues expires. Whether the model reads them is
a property of the model, not the harness; see `deploy/inference/README.md`.

Reading a reply whose author turned its ping off needs the privileged Message Content intent, which
is enabled on the application at discord.com/developers. Without it Discord withholds the text and
the reply goes unanswered.

A turn is one message, edited in place. It opens with a spinner header and gains one line per progress event. The engine writes the text of each line; the bot renders it and never keeps its own copy of the event enum. A line that carries a `slot` writes over the earlier line in that slot, so a tool result replaces its own start line. A `clear` event removes a line. A `draft` event sets or withdraws the answer shown under the steps. Edits are paced to one per `bot.edit_every_ms`, and a change inside the gap waits for the next edit.

The finished card shows the steps (the oldest fold into a count when space runs out), the answer, the memories the prompt carried (`ChatResponse.memories`), and sources without a URL. Sources with a URL are link buttons. Only an answer that still does not fit spills into further messages. An approval adds buttons to the card, and the resumed answer replaces the prompt on the same card.

A conversation belongs to one tenant, user, channel, and visibility. The bot holds no conversation state. `/reset` ends the caller's open conversations in the channel through `POST /conversation/reset`.

## Storage

| Data | Store |
|---|---|
| users, conversations, messages, memories, profile graph, confirmations | PostgreSQL |
| sources, source versions, query source registry, jobs | PostgreSQL |
| chunk text, `vector(1024)` embedding, generated `tsvector` | PostgreSQL with pgvector, rebuildable from snapshots |
| raw page snapshots | object storage |
| local traces, console logs, training data | `.sparky/`, ignored |

```mermaid
erDiagram
    users ||--o{ conversations : opens
    conversations ||--o{ messages : contains
    users ||--o{ memories : owns
    messages ||--o| memories : sources
    users ||--o{ confirmations : approves
    users ||--o{ profile_nodes : describes
    profile_nodes ||--o{ profile_edges : relates
    sources ||--o{ source_versions : "versioned by"
    sources ||--o{ chunks : "chunked into"
    source_versions ||--o{ chunks : produced
    chunks ||--o{ chunks : summarizes
```

`jobs` and `query_sources` stand alone. HNSW, GIN, and tenant, category, and fetch-time indexes serve retrieval. Redis holds the live query cache: answers within their lifetime and the leases that keep one fetch per query. It is shared ephemeral state, so several engine replicas coalesce against each other rather than each fetching once.

## Failure behavior

| Failure | Response |
|---|---|
| model returns a retryable error | retry with backoff up to `agent.max_model_retries` within the deadline, then 502 |
| all model slots busy past `agent.model_queue_wait_secs` | 503, which the bot reports as temporary |
| model call outlives the request deadline | run ends as `Deadline` |
| tool argument error or scraper refusal | the reason becomes the tool result and the model corrects the call |
| tool timeout | `ToolError::Timeout` fed back as the tool result |
| scraper not running | unclaimed job cancelled after `query.claim_secs`, the tool says so |
| model repeats an identical tool call | repeat refused, next step offers no tools, `Stalled` if it repeats again |
| thinking spends the whole completion | the call is made again without thinking |
| prompt would exceed the budget | evidence and history are trimmed, the boot checks keep tool schemas under half |
| retrieval returns nothing | the prompt tells the model to search or say it does not know |
| guardrail blocks a response | run ends as `Blocked` with `guardrail.replacement` |
| confirmation denied, expired, or answered by someone else | nothing runs |
| PostgreSQL unavailable | the engine does not boot, and a request that needs a store gets 503 |
| JSONL trace over `trace.max_file_bytes` | later events for that request are dropped |

## Tracing

Every request produces one trace covering model calls, retrieval, memory recall, tool calls, policy decisions, guardrail blocks, and the outcome. The live pieces of a streaming call (reasoning so far, answer drafts, a withdrawn draft) go only to the watcher. The recorded trace keeps the finished call.

```mermaid
flowchart TD
    A["discord.ask<br/>bot"] --> B["http.chat<br/>engine, parented by traceparent"]
    B --> C["agent.run<br/>invoke_agent"]
    C --> D["retrieve<br/>chunks returned"]
    C --> E["llm<br/>full prompt and reply, thinking"]
    C --> F["tool<br/>redacted arguments and result"]
    F --> G["scrape.query<br/>scraper, separate trace"]
    E -.->|"one llm span, one training example"| H["apps/training data export"]
```

Traces go to two places:

- JSONL at `.sparky/traces/<request_id>.jsonl`: the complete local record of trace events.
- Phoenix: spans exported over OTLP/HTTP to `telemetry.phoenix_url` plus `/v1/traces`, read as a tree per conversation. They carry `gen_ai.*` attributes beside the OpenInference ones the Phoenix UI reads: `user.id` for the user, `session.id` for the conversation. The resource attribute `openinference.project.name` puts them in the project named by `telemetry.project_name`. `apps/training` reads `llm` spans back through `GET /v1/projects/<project>/spans` to build datasets.

An empty `phoenix_url` turns export off. `telemetry.phoenix_api_key` is sent as a bearer token when it is set, and the local Phoenix needs none. The scraper exports `scrape.source`, `scrape.index`, and `scrape.query` spans. The bot records each product event as one span under the interaction (`[analytics]`).

Secrets, credentials, cookies, and sensitive form values are redacted from tool arguments and results before they reach any trace. The developer console mirrors followed stdout into `.sparky/logs/<unit>.log`. Deployments keep stdout with the platform log driver.

## Metrics

`llama-server` runs with `--metrics` and exports Prometheus format on its own port. Prometheus scrapes it and Grafana reads Prometheus. Both are opt-in (`just metrics`) and bind to loopback.

```mermaid
flowchart LR
    subgraph inference ["llama-server"]
        CH["chat :8000/metrics"]
        EM["embed :8001/metrics"]
    end
    GX["gpu-exporter :9835<br/>nvidia-smi, gpu-metrics profile"]
    CH --> P[("prometheus :9090<br/>15s scrape, 15d retention")]
    EM --> P
    GX --> P
    P --> G["grafana :3000<br/>throughput, queue, batching"]
    EN["engine"] -->|"inference"| CH
    EN -->|"OTLP spans"| PX["phoenix :6006<br/>prompt, reply, tokens, latency"]
```

Phoenix holds spans and events. Prometheus holds time series. A slow request shows as the `llm` span's latency in Phoenix and as `llamacpp:requests_deferred` in Grafana for the same minute. Dashboard panels and their metric names are in `deploy/README.md`.

## Authenticated tasks (Phase 8)

Not built. The browser MCP server that Phase 8 assumed is gone, and nothing drives a page. Whatever replaces it needs a new design that meets these rules: the user completes login and MFA themselves, SparkyAI never asks for or stores a password, authenticated page content is never indexed or memorized, and any consequential submission is confirmed by the user.

## Configuration

Two layers, lowest first: `sparky.toml`, then `SPARKY_<SECTION>__<KEY>` environment variables, which win. `sparky.toml` is committed and holds every tunable value, so a change to retrieval fusion or a prompt budget is reviewed like code. `.env` is not committed and holds only secrets, per-machine URLs, and what docker compose and the justfile read. Both images copy `sparky.toml` in, and compose overrides the service URLs through the environment.

Rust reads the file with figment, Python with tomllib through pydantic-settings. `SPARKY_CONFIG_FILE` points at a different file, which is how an eval profile differs from the default. A missing file is not an error.

Sections in `sparky.toml`: `app`, `agent` (with `agent.thinking`), `prompt`, `model` (with `model.sampling`), `embedding`, `summary`, `retrieval` (with `retrieval.router`), `policy`, `tools`, `profile` (with `profile.detector`), `sandbox`, `guardrail`, `compaction`, `query` (with `query.cache`), `mcp`, `trace`, `telemetry`, `analytics`, `http`, `bot`, `postgres`, `scraper`, `search`, `firecrawl`, `object_store`, `cli`, `training`. The engine's `engine` and `discord` sections hold only env values: the service token and the guild id.

A default belongs to exactly one settings struct. Adapters build themselves from those structs and declare no defaults of their own. `Config::validate` rejects at boot any combination the engine cannot serve, for example both retrieval legs off, a section budget above the prompt budget, a sample ratio out of range, two MCP servers with the same name, a text search configuration that is not a plain identifier, or a zero query poll interval. A `prompt.system_file` that cannot be read also stops the boot. Nothing is clamped at runtime.

`prompt` holds the wording the harness writes around every section. `system_file` takes precedence over `system`, which takes precedence over the built-in prompt.

## Deployment

Two images: `sparkyai-rust` (engine and discord, selected by entrypoint) and `sparkyai-scraper` (runs `serve`). CD rebuilds only the images whose inputs changed. Datastores run beside them in Compose. `llama-server` runs as the compose services `chat` and `embed` under the `model` profile, configured in `deploy/inference`. Split further only on a measured need: independent scaling, failure isolation, hardware, or a security boundary. Details are in `deploy/README.md`.

## Open decisions

- Chat model size and quantization.
- Whether a reranker earns its place once the eval set exists.
- Parallel slots per `llama-server` under load.
- Default `chars_per_token` once the tokenizer is measured.
- Retention periods for memories and traces.
- Moderator access to user conversations and traces.
- App server host.

Record each in the commit that settles it.
