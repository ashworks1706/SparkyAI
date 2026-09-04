# apps/cli — `sparky`

The developer console. One terminal for everything in the repo: start and stop each unit, tail
its output, run tasks, watch the engine's health, and talk to the agent.

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
`:roles mod,admin`, `:reset`, `:clear`, `:help`, `:q`. Anything else is passed to `just`, so
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

Container state comes from `docker compose ps` every three seconds. The status bar probes the
engine's `/health`, the chat server's `/models`, and Phoenix every five seconds. Chat posts to the
engine's `/chat` as user `sparky-cli` and shows the request id so the trace is easy to find in
Phoenix.

## Settings

`SPARKY_ENGINE__BASE_URL` (default `http://localhost:8080`), `SPARKY_ENGINE__SERVICE_TOKEN`,
`SPARKY_MODEL__BASE_URL` (`http://localhost:8000/v1`), `SPARKY_CLI__PHOENIX_URL`
(`http://localhost:6006`), `SPARKY_CLI__LOG_LINES` (5000), `SPARKY_CLI__ROLES`,
`SPARKY_CLI__HEALTH_INTERVAL_SECS` (5). The repo's `.env` is loaded if present.

Layout follows the repo convention: `core/{config,types,tests}` hold data and settings;
`runner`, `logs`, `health`, `chat`, `app`, `ui`, `units` hold the objects that do the work.
