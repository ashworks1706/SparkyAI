# SparkyAI v2 Roadmap

## 3 — Discord v0.3

- [ ] Add moderator ops: tickets, announcements, polls, escalation
- [ ] First deployment

## 4 — Memory, MCP, admin v0.4

- [ ] Give the agent memory across conversations, with user-visible control and deletion
- [ ] Personalized discovery and deadlines
- [ ] Admin surface: tools, sources, instructions, limits, trace inspection, approvals, rollback

## 5 — Public beta v0.5

- [ ] Staging + prod; canary releases; alerting; inference that scales under load

## 6 — sparky-model-v0.1

Only after Phase 4 yields clean interaction data.

- [ ] Baseline the untouched model on the eval suite
- [ ] Post-train in stages, each gated on evals (`just train sft` + `just eval compare`; pipeline in place, no run yet)
- [ ] Release: weights, quantized variants, config, dataset description, evals, limitations

## 7 — Sandboxed automation v0.6

- [ ] Authenticated browser sessions through the Playwright MCP server: one isolated context per user, allowlisted domains, limits, logging, cleanup
- [ ] Human confirmation for any authenticated or consequential submission

## 8 — v1.0

- [ ] Stable API, documented traits, published evals, university-adapter template

## Ingestion depth

Not sequenced yet. Retrieval answers from whatever the scraper wrote, so answer quality is
capped here.

- [ ] Crawl past the landing page: courses, clubs, jobs and scholarships each index one navigation page
- [ ] Per-source extraction, so tables, dates and structured fields survive the shared HTML heuristic
- [ ] Chunk on document structure instead of fixed character windows
- [ ] Deduplicate across sources, not only against a source's own previous version
- [ ] Carry provenance a citation can use: section, effective dates, and what supersedes what
- [ ] A quality floor per source: refuse a fetch that yields materially less than the last good one
- [ ] Refresh on the interval each source declares; `last_fetch` only advances when content changed, so every source is due on every poll
- [ ] A fixture per source with the fields it must yield, so an extractor change fails in CI rather than in an answer

## Out of scope until stated otherwise

MyASU / authenticated integrations · GPA or coursework access · unrestricted browser autonomy · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.
