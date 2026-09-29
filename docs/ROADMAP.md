# SparkyAI v2 Roadmap

## 3 — Ingestion and answers v0.3

- [ ] Crawl past the landing page on scheduled runs: courses, clubs, jobs and scholarships each index one navigation page. Pages a live search fetches are already indexed
- [ ] Chunk on document structure instead of fixed character windows
- [ ] have better filtered and document block storage in vector db index
- [ ] Measure whether the tree earns its model calls, once the eval suite exists
- [ ] Deduplicate across sources, not only against a source's own previous version
- [ ] Carry provenance a citation can use: section, effective dates, and what supersedes what
- [ ] First deployment

## 5 — Memory v0.5

- [ ] Personalized discovery and deadlines
- [ ] Admin surface: tools, sources, instructions, limits, trace inspection, approvals, rollback

## 6 — Public beta v0.6

- [ ] run on existing benchmarks
- [ ] setup evals
- [ ] make short tech writeup on readme
- [ ] add demo video and website update
- [ ] Staging + prod; canary releases; alerting; inference that scales under load

## 7 — sparky-model-v0.1

Only after Phase 5 yields clean interaction data.

- [ ] Baseline the untouched model on the eval suite
- [ ] Post-train in stages, each gated on evals (`loupe train sft` in [loupe](https://github.com/ashworks1706/loupe), then `just eval compare` here; pipeline in place, no run yet)
- [ ] Release: weights, quantized variants, config, dataset description, evals, limitations

## 8 — Authenticated tasks v0.7

- [x] A mechanism for per-user authenticated sessions. Grants are stored per user in `oauth_grants` behind `OAuthStore`; `/login <service>` and the `/oauth/{provider}/{authorize,callback,logout}` routes drive the OAuth 2.0 flow over a provider map, the user signs in through the provider themselves, and no password is stored.
- [ ] Human confirmation for any authenticated or consequential submission
- [x] Add Canvas, MyASU integration. Read-only Canvas per user (courses, assignments, grades, announcements, calendar, per-assignment grades), direct-message only, on with `[oauth.canvas]` plus `[canvas]`. MyASU integrations beyond Canvas are still open.
- [x] Add Microsoft/Outlook. Read-only Outlook calendar and mail per user, direct-message only, on with `[oauth.microsoft]` plus `[outlook]`. Needs an Azure app registration and ASU admin consent to reach ASU accounts.
- [x] Add Google Calendar. Native read-only `google_calendar` tool (`runtime/tools/gcal`) per user, direct-message only, on with `[oauth.google]` plus `[gcal]`. Needs a Google Cloud OAuth client requesting `calendar.events.readonly`. Built native rather than over the MCP server, which cannot inject a per-user token.
- [x] Public no-auth tools: `search_papers` (Semantic Scholar), `wikipedia_lookup`, and `valley_metro` (GTFS-realtime, off until a feed URL is set).
- [x] Response guardrail: block denied phrases, redact registered tool names from answers, and defend identity and the prompt against disclosure and injection in the system prompt. Guardrail blocks and redactions are traced and shown on the `agent.run` span.

## 9 — v1.0

- [ ] Stable API, documented traits, published evals, university-adapter template

## Out of scope until stated otherwise

GPA or coursework access · browser automation of any kind · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.
