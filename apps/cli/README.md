# apps/cli: `sparky`

The developer console. It starts services, runs tasks, shows health, and tails output.

```bash
just cli          # from anywhere inside the repo; requires the admin ASU session
```

To chat with the agent, point any OpenAI-compatible client at the engine: base URL `http://localhost:8080/v1`, model `sparky`, API key `SPARKY_ENGINE__SERVICE_TOKEN`. llama-server serves its own web UI for the raw model on http://localhost:8000.

```
 sparky  NORMAL  ● engine ● model ● phoenix                        started engine
╭ units ─────────────────╮╭ engine · running · 42s · 318 lines · follow ───────────╮
│ infra                  ││ 12:01:07 $ setsid just engine                          │
│  ● postgres     :5432  ││ 12:01:09    Compiling engine v0.1.0                    │
│  ● redis        :6379  ││ 12:01:31 INFO engine: listening addr=0.0.0.0:8080      │
│  ○ minio        :9001  ││ 12:01:40 INFO engine::routes::chat: request_id=…       │
│  ● phoenix      :6006  ││ 12:01:44 INFO engine::runtime: tool search_live 412 ms │
│  ● prometheus   :9090  ││ 12:01:47 INFO engine::routes::chat: answered 2 steps   │
│  ● grafana      :3000  ││                                                        │
│ models                 ││                                                        │
│  ● chat         :8000  ││                                                        │
│  ● embed        :8001  ││                                                        │
│ tools                  ││                                                        │
│  ○ searxng      :8888  ││                                                        │
│  ○ firecrawl    :3002  ││                                                        │
│ apps                   ││                                                        │
│  ● engine       :8080  ││                                                        │
│  ○ discord             ││                                                        │
│  ○ scraper             ││                                                        │
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

Commands: `:start engine`, `:stop chat`, `:restart discord`, `:clear`, `:help`, `:q`. Anything else is passed to `just`, so `:eval run` or `:scraper run events` runs as a task under **tasks**.

## What it drives

| Group | Units | How |
|---|---|---|
| infra | postgres, redis, minio, phoenix, pgweb, prometheus, grafana, gpu-exporter | `docker compose up -d` / `stop`, logs via `compose logs -f`; `--profile phoenix` / `db` / `metrics` / `gpu-metrics` where gated |
| models | chat, embed | same, `--profile model` |
| tools | searxng, firecrawl | same, `--profile search` / `crawl` |
| apps | engine, discord, scraper, web | `setsid just <recipe>`; stop sends SIGTERM to the process group |
| sandboxes | the switch, then one row per live container | read from the engine at `GET /sandbox`. Enter on the switch stops the agent being offered `run_sandbox`; enter on a container removes it. The switch's log pane lists every command the agent ran, the running one included |
| tasks | doctor, setup, migrate, scraper status, check, scraper run --all, data export, eval run, eval compare, train sft | `just <recipe>`, exit code shown as ✓ / ✗. Other recipes run from the command line and appear here while they run |
| deploy | up, down, ps, logs, images, prod-up, prod-down, prod-logs | the same recipes as a deploy host; `prod-*` pull GHCR images tagged `SPARKY_IMAGE_TAG` |

Container state comes from `docker compose ps`. The status bar probes the engine, the chat model, and Phoenix.

## Settings

`SPARKY_ENGINE__BASE_URL`, `SPARKY_ENGINE__SERVICE_TOKEN` (for the sandbox routes), `SPARKY_MODEL__BASE_URL`, `SPARKY_CLI__PHOENIX_URL`, `SPARKY_CLI__LOG_LINES`, `SPARKY_CLI__LOG_DIR`, and `SPARKY_CLI__HEALTH_INTERVAL_SECS`. Defaults are in `src/core/config.rs`.

The console keeps a bounded in-memory view and appends full unit output to `.sparky/logs/`, rotated at `cli.log_file_max_mb`.
