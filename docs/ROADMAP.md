# SparkyAI v2 Roadmap

## 3 — Ingestion and answers v0.3

- [ ] Crawl past the landing page: courses, clubs, jobs and scholarships each index one navigation page
- [ ] Chunk on document structure instead of fixed character windows
- [ ] Deduplicate across sources, not only against a source's own previous version
- [ ] Carry provenance a citation can use: section, effective dates, and what supersedes what
- [ ] A quality floor per source: refuse a fetch that yields materially less than the last good one
- [ ] Refresh on the interval each source declares; `last_fetch` only advances when content changed, so every source is due on every poll
- [ ] First deployment

## 4 — Harness shape v0.4

- [ ] Cite only the evidence that entered the prompt; today every retrieved chunk is cited and tool-found sources are not
- [ ] Redact tool output the way arguments already are, before authenticated pages reach a trace

## 5 — Memory and skills v0.5

- [ ] Sandbox sessions: an id that resumes, rather than a fresh container per command
- [ ] Recall a relation, not only a node: edges are written and never read back
- [ ] Skills promoted from a trace, with review before one is offered
- [ ] Give the agent memory across conversations, with user-visible control and deletion
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

- [ ] Authenticated browser sessions through the Playwright MCP server: one isolated context per user, allowlisted domains, limits, logging, cleanup
- [ ] Human confirmation for any authenticated or consequential submission
- [ ] MyASU: the first authenticated integration, read-only before anything else

## 9 — v1.0

- [ ] Stable API, documented traits, published evals, university-adapter template

## Out of scope until stated otherwise

GPA or coursework access · unrestricted browser autonomy · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.

MyASU moved into phase 8 on 2026-09-10; see `decisions/0001-myasu.md`. Phases renumbered the same day when the harness diagram split phase 4 into harness shape and memory.
