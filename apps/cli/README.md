# apps/cli — `sparky`

The developer console starts services, runs tasks, shows health, and tails output.

To chat with the agent, point any OpenAI-compatible client at the engine: base URL
`http://localhost:8080/v1`, model `sparky`. That drives the whole loop — retrieval, tools,
policy, citations. For the raw model with no agent around it, llama-server serves its own web UI
on http://localhost:8000.

```bash
just cli          # from anywhere inside the repo
```

```
 sparky  NORMAL  ● engine ● model ● phoenix                        started engine
╭ units ─────────────────╮╭ engine · running · 42s · 318 lines · follow ───────────╮
│ infra                  ││ 12:01:07 $ setsid just engine                          │
│  ● postgres     :5432  ││ 12:01:09    Compiling engine v0.1.0                    │
│  ● redis        :6379  ││ 12:01:31 INFO engine: listening addr=0.0.0.0:8080      │
│  ○ minio        :9001  ││ 12:01:40 INFO engine::routes::chat: request_id=…       │
│  ● phoenix      :6006  ││ 12:01:44 INFO engine::agent: tool public_search 412 ms │
│  ● prometheus   :9090  ││ 12:01:47 INFO engine::routes::chat: answered 2 steps   │
│  ● grafana      :3000  ││                                                        │
│ models                 ││                                                        │
│  ● chat         :8000  ││                                                        │
│  ● embed        :8001  ││                                                        │
│ tools                  ││                                                        │
│  ○ firecrawl    :3002  ││                                                        │
│  ○ playwright-mcp :8931││                                                        │
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
| `:` | command line, see below |
| `q` | quit; host processes are stopped, containers stay up |

Commands: `:start engine`, `:stop chat`, `:restart discord`, `:clear`, `:help`, `:q`. Anything
else is passed to `just`, so `:eval run` or `:scraper run events` runs as a task and shows up
under **tasks**.

## What it drives

| Group | Units | How |
|---|---|---|
| infra | postgres, redis, minio, phoenix, prometheus, grafana, gpu-exporter | `docker compose up -d` / `stop`, logs via `compose logs -f` |
| models | chat, embed | same, `--profile model` |
| tools | firecrawl, playwright-mcp | same, `--profile crawl` / `browser` |
| apps | engine, discord, web | `setsid just <recipe>`; stop sends SIGTERM to the process group so `cargo run` and its binary both go |
| tasks | doctor, setup, migrate, scraper run, check, data *, eval *, train sft | `just <recipe>`, exit code shown as ✓ / ✗ |
| deploy | up, down, ps, logs, images, prod-up, prod-down, prod-logs | the same recipes the RunPod host uses; `prod-*` pull GHCR images tagged `SPARKY_IMAGE_TAG` |

Container state comes from `docker compose ps`. The status bar probes the engine, chat model, and Phoenix.

## Settings

`SPARKY_ENGINE__BASE_URL`, `SPARKY_MODEL__BASE_URL`, `SPARKY_CLI__PHOENIX_URL`, `SPARKY_CLI__LOG_LINES`, `SPARKY_CLI__LOG_DIR`, and `SPARKY_CLI__HEALTH_INTERVAL_SECS`. Defaults are in `.env.example` and `src/core/config.rs`.

The console keeps a bounded in-memory view and appends full unit output to `.sparky/logs/`.
