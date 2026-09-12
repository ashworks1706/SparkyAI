# SparkyAI v2 Roadmap

## 3 — Ingestion and answers v0.3

- [ ] Crawl past the landing page: courses, clubs, jobs and scholarships each index one navigation page
- [ ] Chunk on document structure instead of fixed character windows
- [ ] Measure whether the tree earns its model calls, once the eval suite exists
- [ ] Deduplicate across sources, not only against a source's own previous version
- [ ] Carry provenance a citation can use: section, effective dates, and what supersedes what
- [ ] First deployment

## 5 — Memory and skills v0.5

- [ ] Skills promoted from a trace, with review before one is offered
- [ ] Train the fact detector on real turns; the gate is rules until there is labelled data
- [ ] Personalized discovery and deadlines
- [ ] Moderator ops: tickets, announcements, polls, escalation
- [ ] Admin surface: tools, sources, instructions, limits, trace inspection, approvals, rollback

## 6 — Public beta v0.6

- [ ] Staging + prod; canary releases; alerting; inference that scales under load

## 7 — sparky-model-v0.1

Only after Phase 5 yields clean interaction data.

- [ ] Baseline the untouched model on the eval suite
- [ ] Post-train in stages, each gated on evals (`just train sft` + `just eval compare`; pipeline in place, no run yet)
- [ ] Release: weights, quantized variants, config, dataset description, evals, limitations

## 8 — Authenticated tasks v0.7

- [ ] A mechanism for authenticated sessions. The browser MCP server is gone (decisions/0005); this needs a decision note before any work
- [ ] Human confirmation for any authenticated or consequential submission
- [ ] MyASU: the first authenticated integration, read-only before anything else

## 9 — v1.0

- [ ] Stable API, documented traits, published evals, university-adapter template

## Out of scope until stated otherwise

GPA or coursework access · browser automation of any kind · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.
