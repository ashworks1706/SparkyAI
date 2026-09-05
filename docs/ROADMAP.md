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

## Out of scope until stated otherwise

MyASU / authenticated integrations · GPA or coursework access · unrestricted browser autonomy · university-wide deployment · FERPA claims · one agent per domain · RL before evals exist.
