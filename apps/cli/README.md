# apps/cli — `sparky`

The developer console starts services, runs tasks, shows health, tails output, and chats with the engine.

```bash
just cli          # from anywhere inside the repo
```

```
 sparky  NORMAL  ● engine ● model ● phoenix                          started engine
╭ units ─────────────────╮╭ engine · running · 42s · 318 lines · follow ─────────────╮
│ infra                  ││ 12:01:07 $ setsid just engine                              │
│  ● postgres     :5432  ││ 12:01:09    Compiling engine v0.1.0                        │
│  ● redis        :6379  ││ 12:01:31 INFO engine: listening addr=0.0.0.0:8080          │
│  ○ minio        :9001  ││ 12:01:40 INFO engine::routes::chat: request_id=…            │
│  ● phoenix      :6006  ││                                                            │
│ models                 │╰──────────────────────── the agent and its HTTP surface ────╯
│  ● chat         :8000  │╭ chat · 7c0e… · roles [] ─────────────────────────────────╮
│  ● embed        :8001  ││ you                                                        │
│ tools                  ││   what are hayden library's hours today?                   │
│  ○ firecrawl    :3002  ││ sparky                                                     │
│  ○ playwright-mcp :8931││   Hayden Library is open 7am–2am today [1].                │
│ apps                   ││   ↳ library_hours — https://lib.asu.edu/hours              │
│  ● engine       :8080  ││   answered · 2 steps · 1841 tokens · 3120 ms · trace 3a…   │
│  ○ discord             │╰────────────────────────────────────────────────────────────╯
│  ○ web          :5173  │╭ ask ──────────────────────────────────────────────────────╮
│ tasks                  ││ > does it close earlier on friday█                          │
│  ✓ check               │╰────────────────────────────────────────────────────────────╯
│ deploy                 │
│  ○ prod-up             │
 j/k move ⏎ start/stop r restart l/h logs/units / search i chat : command ? help q quit
```

## Keys

Modal, like vim. `?` shows this in the console.

| Key | Does |
|---|---|
| `j` `k` `gg` `G` | move, or scroll the focused pane; `G` on logs resumes following |
| `enter` / `s`, `x`, `r` | start or stop, stop, restart the selected unit |
| `h` `l` `tab` | focus units / logs / chat |
| `/` then `n` `N` | search the selected unit's logs |
| `c`, `i` | toggle the chat pane, type to the agent (`esc` leaves) |
| `R`, `C`, `o` | new conversation, clear logs, open the unit's URL |
| `:` | command line, see below |
| `q` | quit; host processes are stopped, containers stay up |

Commands: `:start engine`, `:stop chat`, `:restart discord`, `:ask <question>`,
`:roles MANAGE_GUILD`, `:reset`, `:clear`, `:help`, `:q`. Anything else is passed to `just`, so
`:eval run` or `:scraper run events` runs as a task and shows up under **tasks**.

## What it drives

| Group | Units | How |
|---|---|---|
| infra | postgres, redis, minio, phoenix | `docker compose up -d` / `stop`, logs via `compose logs -f` |
| models | chat, embed | same, `--profile model` |
| tools | firecrawl, playwright-mcp | same, `--profile crawl` / `browser` |
| apps | engine, discord, web | `setsid just <recipe>`; stop sends SIGTERM to the process group so `cargo run` and its binary both go |
| tasks | doctor, setup, migrate, scraper run, check, data *, eval *, train sft | `just <recipe>`, exit code shown as ✓ / ✗ |
| deploy | up, down, ps, logs, images, prod-up, prod-down, prod-logs | the same recipes the RunPod host uses; `prod-*` pull GHCR images tagged `SPARKY_IMAGE_TAG` |

Container state comes from `docker compose ps`. The status bar probes the engine, chat model, and Phoenix. Chat uses the engine's `/chat` endpoint as `sparky-cli`.

## Settings

`SPARKY_ENGINE__BASE_URL`, `SPARKY_ENGINE__SERVICE_TOKEN`, `SPARKY_MODEL__BASE_URL`, `SPARKY_CLI__PHOENIX_URL`, `SPARKY_CLI__LOG_LINES`, `SPARKY_CLI__LOG_DIR`, `SPARKY_CLI__ROLES`, and `SPARKY_CLI__HEALTH_INTERVAL_SECS`. Defaults are in `.env.example` and `src/core/config.rs`.

The console keeps a bounded in-memory view and appends full unit output to `.sparky/logs/`.
