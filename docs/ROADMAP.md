# SparkyAI v2 Roadmap

Ground-up rebuild. v1 is preserved on `archive/v1` and the `v1.0-original` release.

## Projects

| Project | What | Language |
|---|---|---|
| **SparkyAI** | Discord (later web) copilot for ASU students and AI Society moderators: engine, bot, scraper, web | Rust · Python · TypeScript |
| **Sparky Models** | Post-trained open models for routing and tool use | Python (PyTorch, TRL, PEFT) |

The agent harness is a module of the engine, not a separate product.

## Principles

- Open models only. Qwen baseline, served via vLLM's OpenAI-compatible API. Hosted models are judges/teachers, never the foundation.
- Facts live in retrieval, not weights. Models learn behavior.
- Never scrape at query time. Ingestion is offline.
- Per-request context. No global state.
- Trait per replaceable dependency in `engine/src/agent/harness`: `ModelProvider`, `Tool`, `Retriever`, `MemoryStore`, `ConversationStore`, `Policy`, `TraceSink`, `Sandbox`.
- Evals before training.

## Stack

| Layer | Choice |
|---|---|
| Engine / Discord | Rust — tokio, axum, serenity, serde, sqlx |
| LLM client / tools / embeddings | Rig (`rig-core`); our loop runs on its `CompletionModel`, not its `Agent` |
| MCP | `rmcp` (official SDK) |
| Web | Vite + React + TypeScript + shadcn (`apps/web`); landing now, admin UI in Phase 4 |
| Model serving | vLLM (OpenAI-compatible HTTP) |
| Chat model | Qwen3-14B on vLLM (RunPod) |
| Embeddings / reranker | Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B on vLLM |
| Database | PostgreSQL |
| Cache / queue | Redis |
| Vector store | Qdrant (adapter); Piramid later |
| Memory / knowledge layer | Own implementation on Postgres + Qdrant |
| Scraper | Python worker (`apps/scraper`): httpx + BeautifulSoup, Playwright where JS is required |
| Object storage | S3-compatible (MinIO locally) |
| Post-training | Python: TRL + PEFT (+ Unsloth), W&B, HF Hub |
| Evals | Inspect AI, BFCL, lm-eval |
| Observability | Sentry (errors), OpenTelemetry → Axiom (traces), JSON logs |
| Deploy | Docker Compose; GHCR images via CD; RunPod GPU pods for vLLM |

## Phases

### 0 — Archive, clean, scaffold
1. Tag `v1.0-original`, branch `archive/v1`, GitHub release with v1 screenshots.
2. On `main`, remove v1: all Python source, `finetune/`, `tests/` screenshots, Docker files, CI, `requirements.txt`. Keep `README.md`, `LICENSE`, `docs/`.
3. Monorepo layout: `apps/{engine,discord}` (Rust bins), `apps/{scraper,training}` (Python), `apps/inference` (vLLM config), `apps/web`, `apps/sandbox`, `deploy`. See ARCHITECTURE.md.
4. `docs/ARCHITECTURE.md`: request lifecycle, crate boundaries, trait list.
5. CI: `cargo fmt`, `cargo clippy`, `cargo test`.

**Exit:** every unit builds and lints in CI; v1 is one `git checkout archive/v1` away.

### 1 — Harness module v0.1
- Message / tool-call / tool-result types
- Rig `CompletionModel` against vLLM; mock model for tests
- `Tool` = Rig `Tool` + `RiskClass` + `RequestContext`; adapter so any Rig tool drops in
- Agent loop: step limit, timeouts, retries, cancellation, parallel tool calls
- `RequestContext` threaded through everything
- `TraceSink` trait + JSONL; replay from trace
- Token/cost accounting; mock model; tests

**Exit:** clone → one command → local agent, one tool, one conversation, trace file, passing tests.

### 2 — Retrieval v0.2
- `Retriever` trait; Qdrant adapter; metadata + permissions in Postgres
- Ingestion jobs: fetch → timestamp → extract → dedupe → chunk → embed → index → change detection
- Port v1 sources, simplest first: library hours, events, clubs, courses, scholarships, news, shuttles, jobs, sports
- Hybrid retrieval, rerank, citations, freshness
- Fixed ASU eval set

**Exit:** correct, dated sources on the eval set; reproducible traces.

### 3 — Discord v0.3
- `POST /chat` + SSE on `engine`; `discord` as a thin HTTP client of it
- Conversation state in Postgres; Discord identity + role checks in `Policy`
- Moderator ops with confirmation before any write: tickets, announcements, polls, escalation
- First deployment

**Exit:** AI Society uses it daily; failures inspectable from traces.

### 4 — Memory, MCP, admin v0.4
- `MemoryStore` trait; memory kinds: working, conversation, episodic, semantic, profile
- Write policy (useful, stable, sensitive, approved, expiry); user-visible memory with deletion
- Personalized discovery and deadlines
- MCP client via `rmcp`
- Admin: tools, sources, instructions, limits, trace inspection, approvals, rollback
- Eval suites: tool selection/args, grounding, memory, permissions, clarification, refusal, latency

### 5 — Public beta v0.5
- Staging + prod; vLLM workers behind a queue; canary releases; alerting

### 6 — sparky-model-v0.1
Only after Phase 4 yields clean interaction data.
1. Baseline untouched Qwen on the eval suite
2. Dataset: tool use, retrieval queries, clarification, permission boundaries, trajectories, adversarial cases. PII removed. v1 `finetune/` is regenerated, not reused.
3. SFT (LoRA) → 4. DPO → 5. GRPO on verifiable rewards only
Release: weights, quantized variants, config, dataset description, evals, limitations.

### 7 — Sandboxed automation v0.6
- `Sandbox` trait; isolated browser per task; allowlisted domains; credential isolation; limits; action logs; cleanup
- Human confirmation for any authenticated or consequential submission

### 8 — v1.0
Stable API, documented traits, published evals, university-adapter template.

## Out of scope until stated otherwise

MyASU / authenticated integrations · GPA or coursework access · unrestricted browser autonomy · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.

## Sequencing rule

Harness and evals → features → clean data → models.
