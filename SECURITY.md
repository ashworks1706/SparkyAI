# Security

SparkyAI answers from sources it scrapes, calls tools on a user's behalf, and remembers the people who talk to it. Report anything that could expose what it remembers, put content it did not retrieve into an answer, or make it act without permission.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/ashworks1706/SparkyAI/security/advisories/new), not a public issue. Expect an acknowledgement within a week. SparkyAI is pre-1.0; only the latest release and `main` are supported.

## In scope

- one guild or user reaching another's conversations, profile graph or remembered facts
- a token, key or secret reaching a model prompt, a log line, a trace, a metric or a Discord message
- personal memory or the profile graph reaching an answer others can read
- a write-side tool running without its confirmation, or confirmed by someone without the role
- prompt injection through a scraped page or a tool result that leads to an action nobody asked for
- model output written back into the retrieval index as if it were evidence
- the admin ASU session, or content fetched with it, reaching the index, memory, or a trace

## Model

- The tenant is the guild. Conversations, memory and the profile graph are scoped by `tenant_id`.
- A direct message is `Visibility::Private`; a guild thread is `Visibility::Public`. `agent.recall_in_public` keeps personal memory and the profile graph out of a public answer.
- Secrets are `SecretString`, come from the environment only, and are never logged.
- Write-side tools go through `Policy`. `policy.confirm_from` decides what needs a confirmation, which expires; the model never decides whether one is required.
- Only `apps/scraper` writes the retrieval index or fetches pages. Model output is never written back as evidence. Sources behind the admin session set `index = False`.
- `run_sandbox` runs without network, or with `sandbox.egress` through a proxy that refuses private, loopback, link-local and reserved addresses.
- `apps/engine`, `apps/discord` and `apps/cli` never depend on each other (`scripts/check-deps.sh`).
- Each deployment is self-hosted; no telemetry is sent to the project.

## Dependencies

cargo-deny and pip-audit on dependency changes, dependency review on pull requests, secret scanning on every push, Dependabot weekly.
