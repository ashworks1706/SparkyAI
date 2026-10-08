import { CAMPUS, MCP } from "./campus";

/** Terminal colours the console uses. */
export const COLORS = { accent: "#56b6c2", dim: "#5c6370", green: "#98c379", yellow: "#e5c07b", red: "#e06c75", blue: "#61afef", magenta: "#c678dd" };

/** When each beat of the console scene happens. */
export const AT = {
  shell: [0.3, 1.0],
  tui: 1.4,
  colon: 2.2,
  command: [2.35, 3.6],
  run: 3.8,
  indexed: 4.5,
  exited: 4.8,
  climb: 5.2,
  engine: 6.3,
  engineUp: 8.2,
  slash: 8.5,
  search: [8.6, 8.9],
  hit: 9.1,
  discord: [9.9, 10.2],
  discordUp: 11.0,
  scraper: [11.4, 11.7],
  scraperUp: 12.4,
};

/** The task the developer runs from the command line. */
export const TASK = "scraper run --category calendar";

/** The state of a unit in the console. */
export type Status = "stopped" | "starting" | "running" | "done";

/** A row of the unit list: its id, port, hint, and status at time t. */
type Unit = { id: string; port?: string; hint: string; status?: (t: number) => Status };

/** A status that is stopped before start, starting until up, then running. */
const upFrom = (start: number, up: number) => (t: number): Status => (t < start ? "stopped" : t < up ? "starting" : "running");

/** The catalog of apps/cli/src/units/mod.rs, with what bootstrap and this scene start. */
export const GROUPS: [string, Unit[]][] = [
  [
    "infra",
    [
      { id: "postgres", port: ":5432", hint: "pgvector 17; the retrieval index and conversations", status: () => "running" },
      { id: "redis", port: ":6379", hint: "live query cache, its leases and the cap on fetches", status: () => "running" },
      { id: "minio", port: ":9001", hint: "object store", status: () => "running" },
      { id: "phoenix", port: ":6006", hint: "trace UI: one conversation, its tool calls and timings", status: () => "running" },
      { id: "pgweb", port: ":8081", hint: "browse the database in a browser" },
      { id: "prometheus", port: ":9090", hint: "scrapes llama-server metrics" },
      { id: "grafana", port: ":3000", hint: "inference dashboards over Prometheus" },
      { id: "gpu-exporter", port: ":9835", hint: "GPU utilisation and VRAM, needs an NVIDIA GPU" },
    ],
  ],
  [
    "models",
    [
      { id: "chat", port: ":8000", hint: "llama-server chat model", status: () => "running" },
      { id: "embed", port: ":8001", hint: "llama-server embedding model", status: () => "running" },
    ],
  ],
  [
    "tools",
    [
      { id: "searxng", port: ":8888", hint: "self-hosted metasearch behind the web source of search_live", status: () => "running" },
      { id: "firecrawl", port: ":3002", hint: "self-hosted Firecrawl for the scraper", status: () => "running" },
    ],
  ],
  [
    "apps",
    [
      { id: "engine", port: ":8080", hint: "the agent and its HTTP surface", status: upFrom(AT.engine, AT.engineUp) },
      { id: "discord", hint: "serenity bot, HTTP client of the engine", status: upFrom(AT.discord[1], AT.discordUp) },
      { id: "scraper", hint: "live searches, their indexing, and scheduled runs, from one queue", status: upFrom(AT.scraper[1], AT.scraperUp) },
      { id: "web", port: ":5173", hint: "Vite dev server" },
    ],
  ],
  ["sandboxes", [{ id: "sandbox", hint: "what the agent runs; enter stops it being offered", status: () => "running" }]],
  [
    "tasks",
    [
      { id: "doctor", hint: "required tools, .env, hooks" },
      { id: "setup", hint: "install every unit's deps" },
      { id: "migrate", hint: "apply scraper migrations", status: () => "done" },
      { id: "scraper status", hint: "every source, its last run, and the job queue" },
      { id: "check", hint: "fmt, lint, test every unit" },
      { id: "scraper run --all", hint: "every source and static page, paced per host" },
      { id: "eval run", hint: "golden cases against the engine" },
      { id: "eval compare", hint: "fail on regression" },
      { id: TASK, hint: "[one-shot]", status: (t) => (t < AT.exited ? "starting" : "done") },
    ],
  ],
  [
    "deploy",
    [
      { id: "up", hint: "dev stack: build images locally, start everything" },
      { id: "down", hint: "stop the whole stack, every profile" },
      { id: "ps", hint: "what compose has, across every profile" },
    ],
  ],
];

/** One captured log line: when, which stream, and the text. */
type Log = { at: number; text: string; meta?: boolean };

/** What each unit printed in this scene. Engine lines are its tracing events at info. */
export const LOGS: Record<string, Log[]> = {
  [TASK]: [
    { at: AT.run + 0.05, text: `$ just ${TASK}`, meta: true },
    { at: AT.indexed, text: "academic_calendar: indexed (38 chunks)" },
    { at: AT.exited, text: "exited with 0", meta: true },
  ],
  engine: [
    { at: AT.engine + 0.05, text: "$ just engine", meta: true },
    { at: 6.8, text: "INFO engine::wiring: search tools registered, fallback: web" },
    { at: 7.0, text: "INFO engine::wiring: sandbox registered, runtime: docker, image: ghcr.io/ashworks1706/sparkyai-sandbox:main" },
    { at: 7.25, text: `INFO engine::wiring: mcp tools registered, server: ${MCP.name}, url: ${MCP.url}, count: ${MCP.tools.length}` },
    { at: 7.45, text: "INFO engine::wiring: tools registered, integration: canvas, count: 6" },
    { at: 7.6, text: "INFO engine::wiring: tools registered, integration: papers, count: 1" },
    { at: 7.75, text: "INFO engine::wiring: tools registered, integration: wikipedia, count: 1" },
    { at: AT.engineUp, text: "INFO engine::wiring: listening, addr: 0.0.0.0:8080" },
  ],
  discord: [
    { at: AT.discord[1] + 0.05, text: "$ just discord", meta: true },
    { at: AT.discordUp, text: `INFO discord::bot: connected, user: Sparky, guild: ${CAMPUS.guild}` },
  ],
  scraper: [
    { at: AT.scraper[1] + 0.05, text: "$ just scraper serve", meta: true },
    { at: 12.1, text: "[info     ] registry published             sources=18" },
    { at: AT.scraperUp, text: "[info     ] serving                        live=('source_query',) live_workers=4" },
  ],
};

/** The key hints along the bottom line: each key and what it does. */
export const HINTS: [string, string][] = [
  ["j/k", "move"],
  ["⏎", "start/stop"],
  ["r", "restart"],
  ["l/h", "logs/units"],
  ["/", "search"],
  [":", "command"],
  ["o", "open url"],
  ["?", "help"],
  ["q", "quit"],
];
