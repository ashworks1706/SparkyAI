# SparkyAI v2 Roadmap

Ground-up rebuild. v1 is preserved on `archive/v1` and the `v1.0-original` release. Stack and rules: [ARCHITECTURE.md](ARCHITECTURE.md). Phase 0 (archive, clean, scaffold) done 2026-08-28.

## 1 — Harness module v0.1

- [x] Run a full agent conversation against a local model, with a mock for tests
- [x] Let tools plug in with risk levels and per-request context
- [x] Make the agent loop robust: step limits, timeouts, retries, cancellation, parallel tool calls
- [x] Record every run as a trace that can be replayed
- [x] Track token usage and cost

Exit: clone → one command → local agent, one tool, one conversation, trace file, passing tests.
Done 2026-09-04: `just model && just infra && just engine`, then `POST /chat`. Replay of a trace against a chosen model is Phase 2 work alongside the eval set.

## 2 — Retrieval v0.2

- [ ] Stand up search over ASU content with access controls
- [ ] Build ingestion pipelines that keep sources fresh and deduplicated
- [ ] Port v1 sources, simplest first: library hours, events, clubs, courses, scholarships, news, shuttles, jobs, sports
- [ ] Answer with citations and dates; rerank for quality
- [ ] Ship a fixed ASU eval set

Exit: correct, dated sources on the eval set; reproducible traces.

## 3 — Discord v0.3

- [ ] Expose the agent over a streaming chat API; Discord bot as a thin client
- [ ] Persist conversations; enforce Discord identity and role checks
- [ ] Add moderator ops with confirmation before any write: tickets, announcements, polls, escalation
- [ ] First deployment

Exit: AI Society uses it daily; failures inspectable from traces.

## 4 — Memory, MCP, admin v0.4

- [ ] Give the agent memory across conversations, with user-visible control and deletion
- [ ] Personalized discovery and deadlines
- [ ] Connect to external tools over MCP
- [ ] Admin surface: tools, sources, instructions, limits, trace inspection, approvals, rollback
- [ ] Eval suites: tool selection/args, grounding, memory, permissions, clarification, refusal, latency

## 5 — Public beta v0.5

- [ ] Staging + prod; canary releases; alerting; inference that scales under load

## 6 — sparky-model-v0.1

Only after Phase 4 yields clean interaction data.

- [ ] Baseline the untouched model on the eval suite
- [ ] Build a training dataset from real interactions, PII removed; v1 `finetune/` is regenerated, not reused
- [ ] Post-train in stages, each gated on evals
- [ ] Release: weights, quantized variants, config, dataset description, evals, limitations

## 7 — Sandboxed automation v0.6

- [ ] Let the agent drive a browser inside a locked-down sandbox with limits, logging, and cleanup
- [ ] Human confirmation for any authenticated or consequential submission

## 8 — v1.0

- [ ] Stable API, documented traits, published evals, university-adapter template

## Out of scope until stated otherwise

MyASU / authenticated integrations · GPA or coursework access · unrestricted browser autonomy · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.

## Sequencing rule

Harness and evals → features → clean data → models.
