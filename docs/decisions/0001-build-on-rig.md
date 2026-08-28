# 0001 — Build the harness on Rig, not from scratch

**Context.** The harness is a deliverable, but Rust already has a mature LLM layer: Rig (0.6, 20+ providers, MCP, OTel, production users). Writing provider clients, tool schemas, and embedding adapters ourselves adds weeks and no differentiation. LangGraph-style durable-state/interrupt runtimes do not exist in Rust in a form we'd depend on.

**Decision.** Use Rig for `CompletionModel`, `Tool` schema, embeddings, and vector-store adapters. Use `rmcp` for MCP. Write our own agent loop over `CompletionModel` — not `rig::Agent` — so policy, confirmation, step budgets, context assembly, and replayable tracing are ours.

**Consequences.** Harness public API wraps Rig types with `RequestContext` and `RiskClass`; any Rig tool adapts in. We track Rig's breaking changes. What we release is the governed loop + policy + tracing layer, which Rig lacks.
