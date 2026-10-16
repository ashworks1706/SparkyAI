# apps/cli: `sparky`

The developer console. It starts services, runs tasks, shows health, and tails output.

```bash
just cli          # from anywhere inside the repo
```

To chat with the agent, use the Discord bot or `POST /chat` on http://localhost:8080 with the bearer `SPARKY_ENGINE__SERVICE_TOKEN`. llama-server serves its own web UI for the raw model on http://localhost:8000.

```
 sparky  NORMAL  ● engine ● model ● phoenix                        started engine
╭ units ─────────────────╮╭ engine · running · 42s · 318 lines · follow ───────────╮
│ infra                  ││ 12:01:07 $ setsid just engine                          │
│  ● phoenix      :6006  ││ 12:01:09    Compiling engine v0.1.0                    │
│  ● prometheus   :9090  ││ 12:01:31 INFO engine: listening addr=0.0.0.0:8080      │
│  ● grafana      :3000  ││ 12:01:40 INFO engine::routes::chat: request_id=…       │
│ models                 ││ 12:01:47 INFO engine::routes::chat: answered 2 steps   │
│  ● chat         :8000  ││                                                        │
│  ● embed        :8001  ││                                                        │
│ apps                   ││                                                        │
│  ● engine       :8080  ││                                                        │
│  ○ discord             ││                                                        │
│  ○ web          :5173  ││                                                        │
│ tasks                  ││                                                        │
│  ✓ check               ││                                                        │
│ deploy                 ││                                                        │
│  ○ prod-up             │╰─────────────────── the agent and its HTTP surface ─────╯
 j/k move ⏎ start/stop r restart l/h logs/units / search : command o open url ? help q quit
```

## Keys

Modal, like vim. `?` shows this in the console.

| Key | Does |
|---|---|
| `j` `k` `gg` `G` | move, or scroll the focused pane; `G` on logs resumes following |
| `enter` / `s`, `x`, `r` | start or stop, stop, restart the selected unit |
| `h` `l` `tab` | focus units / logs |
| `/` then `n` `N` | search the selected unit's logs |
| `C`, `o` | clear logs, open the unit's URL |
| `:` | command line |
| `q` | quit; host processes stop, containers stay up |

Commands: `:start engine`, `:stop chat`, `:restart discord`, `:clear`, `:help`, `:q`. Anything else is passed to `just`, so `:eval run` or `:sandbox-images` runs as a task under **tasks**.

## What it drives

| Group | Units | How |
|---|---|---|
| infra | phoenix, prometheus, grafana, gpu-exporter | `docker compose up -d` / `stop`, logs via `compose logs -f`; `--profile phoenix` / `metrics` / `gpu-metrics` |
| models | chat, embed | same, `--profile model` |
| apps | engine, discord, web | `setsid just <recipe>`; stop sends SIGTERM to the process group |
| sandboxes | the switch, then one row per live container | read from the engine at `GET /sandbox`. Enter on the switch stops the agent being offered `run_sandbox`; enter on a container removes it. The switch's log pane lists every command the agent ran, the running one included |
| tasks | doctor, setup, check, eval run, eval compare | `just <recipe>`, exit code shown as ✓ / ✗. Other recipes run from the command line and appear here while they run |
| deploy | up, down, ps, logs, images, prod-up, prod-down, prod-logs | the same recipes as a deploy host; `prod-*` pull GHCR images tagged `SPARKY_IMAGE_TAG` |

Container state comes from `docker compose ps`. The status bar probes the engine, the chat model, and Phoenix.

## Settings

`SPARKY_ENGINE__BASE_URL`, `SPARKY_ENGINE__SERVICE_TOKEN` (for the sandbox routes), `SPARKY_MODEL__BASE_URL`, `SPARKY_CLI__PHOENIX_URL`, `SPARKY_CLI__LOG_LINES`, `SPARKY_CLI__LOG_DIR`, and `SPARKY_CLI__HEALTH_INTERVAL_SECS`. Defaults are in `src/core/config.rs`.

The console keeps a bounded in-memory view and appends full unit output to `.sparky/logs/`, rotated at `cli.log_file_max_mb`.
