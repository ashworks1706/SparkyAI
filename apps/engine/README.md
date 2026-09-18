# apps/engine

The agent and the HTTP surface in front of it. One loop: assemble a prompt, call the model, run
the capabilities it asks for, repeat until it answers or hits a limit. Everything else in the
repo is either a client of this or a producer of what it reads.

```bash
just engine                         # needs .env; see CONTRIBUTING for the first run
cargo run -p engine                 # the same thing without just
cargo test -p engine                # unit tests
cargo test -p engine -- --ignored   # the tests that need a container runtime
```

## The surface

| Route | Does |
|---|---|
| `POST /chat` | one turn, answered whole. Bearer `engine.service_token`. |
| `POST /chat/stream` | the same turn as server-sent events: progress, then the answer |
| `POST /confirm` | approves or refuses an action the policy held, by token |
| `POST /conversation/reset` | ends the caller's conversations |
| `POST /profile/forget`, `/profile/list` | what the profile graph holds about the caller |
| `POST /v1/chat/completions`, `GET /v1/models` | OpenAI-compatible, so any client of that API can talk to Sparky |
| `GET /health/live` | the process is up |
| `GET /health/ready` | `{"postgres":bool,"model":bool}`, 503 until both are true |

A model it cannot reach is not a crash: `/health/ready` reports `model:false` and `/chat`
answers 502 `the model is unavailable`.

## Layout

```
src/core/       config, telemetry, types, traits, tests. Imports nothing else in the crate.
src/runtime/    harness (the loop, prompt assembly, memory, safety, compaction, tracing),
                model (the Rig client, the slot limit), tools (search, mcp, sandbox)
src/stores/     postgres and redis: conversations, memory, the retrieval index, the job queue
src/routes/     axum handlers
src/wiring.rs   builds every adapter from config and composes them; the boot checks live here
```

`runtime::harness`, `runtime::model`, `runtime::tools` and `stores` each import only `core`,
never each other. `scripts/check-deps.sh` enforces the part of that rule it can see.

## The loop

`Agent::run` in `runtime/harness/agent/`. Retrieval, memory and history are loaded once before
the first step; later steps grow only by what the model and its tools add. A step calls the
model, passes the response through the guardrail, authorizes the tool calls it asked for
against `Policy`, runs the allowed ones, and continues. It stops on an answer, a confirmation,
`agent.max_steps`, the request deadline, or cancellation.

Tools are `search_knowledge`, `search_live`, `run_sandbox`, and whatever MCP servers are
configured. Each declares a `RiskClass`; anything at
`policy.confirm_from` or above stops the turn and waits for the caller to approve.

## Configuration

Everything tunable is in `sparky.toml` under `agent`, `prompt`, `model`, `retrieval`, `tools`,
`policy`, `sandbox`, `guardrail`, `compaction`, `profile` and `query`. `.env` holds only secrets
and per-machine URLs. `Config::validate` rejects a bad combination at boot rather than clamping
it later, and `wiring.rs` adds the checks that need a live dependency: the prompt against one
model slot, the capabilities against their budget, the sources against what the scraper
publishes, and the container runtime behind `run_sandbox`.
