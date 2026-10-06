import type { ReactNode } from "react";
import { typed } from "../timeline";
import { CAMPUS, MCP } from "./campus";
import { Caret, MacWindow, Prompt, type Scene } from "./kit";

/**
 * The sparky developer console, drawn as apps/cli/src/app/ui.rs draws it:
 * status bar, unit list by group, the selected unit's log pane, and the command line.
 */

/** Terminal colours the console uses. */
const C = { accent: "#56b6c2", dim: "#5c6370", green: "#98c379", yellow: "#e5c07b", red: "#e06c75", blue: "#61afef", magenta: "#c678dd" };

/** When each beat of the console scene happens. */
const AT = {
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
const TASK = "scraper run --category calendar";

/** Seconds per keypress while the selection climbs to the engine. */
const STEP = 0.07;

type Status = "stopped" | "starting" | "running" | "done";

const GLYPH: Record<Status, [string, string]> = {
  stopped: ["○", C.dim],
  starting: ["◐", C.yellow],
  running: ["●", C.green],
  done: ["✓", C.blue],
};

/** A row of the unit list: its id, port, hint, and status at time t. */
type Unit = { id: string; port?: string; hint: string; status?: (t: number) => Status };

const upFrom = (start: number, up: number) => (t: number): Status => (t < start ? "stopped" : t < up ? "starting" : "running");

/** The catalog of apps/cli/src/units/mod.rs, with what bootstrap and this scene start. */
const GROUPS: [string, Unit[]][] = [
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

/** Every unit in sidebar order. */
const ORDER = GROUPS.flatMap(([, units]) => units);

/** The unit selected at time t. */
const selectedAt = (t: number): string => {
  if (t < AT.run) return "postgres";
  if (t < AT.climb) return TASK;
  if (t < AT.discord[0]) {
    const from = ORDER.findIndex((u) => u.id === TASK);
    const to = ORDER.findIndex((u) => u.id === "engine");
    const steps = Math.min(from - to, Math.floor((t - AT.climb) / STEP));
    return ORDER[from - steps].id;
  }
  if (t < AT.scraper[0]) return "discord";
  return "scraper";
};

/** One captured log line: when, which stream, and the text. */
type Log = { at: number; text: string; meta?: boolean };

/** What each unit printed in this scene. Engine lines are its tracing events at info. */
const LOGS: Record<string, Log[]> = {
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

const HINTS: [string, string][] = [
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

/** The wall clock the log pane prints, counting from the scene start. */
const clock = (at: number) => `09:02:${String(10 + Math.floor(at)).padStart(2, "0")}`;

/** A pane with a rounded border and its title on the top edge. */
const pane = ({ title, focused, footer, children }: { title: string; focused: boolean; footer?: string; children: ReactNode }) => (
  <div className="relative h-full rounded-md border px-2 pt-2.5 pb-1" style={{ borderColor: focused ? C.accent : C.dim }}>
    <span className="absolute -top-[12px] left-2 bg-[#16161a] font-bold text-white">{title}</span>
    {footer && (
      <span className="absolute -bottom-[12px] right-2 bg-[#16161a]" style={{ color: C.dim }}>
        {` ${footer} `}
      </span>
    )}
    <div className="h-full overflow-hidden">{children}</div>
  </div>
);

/** The console at scene time t. */
const consoleAt = (t: number) => {
  const selected = selectedAt(t);
  const unit = ORDER.find((u) => u.id === selected) ?? ORDER[0];
  const status = unit.status?.(t) ?? "stopped";
  const logs = (LOGS[selected] ?? []).filter((l) => l.at <= t);
  const command = t >= AT.colon && t < AT.run;
  const searching = t >= AT.slash && t < AT.hit;
  const search = t >= AT.search[0] ? typed("mcp", t, AT.search[0], AT.search[1]) : "";
  const hit = t >= AT.hit ? logs.findIndex((l) => l.text.includes("mcp")) : -1;
  const engineUp = t >= AT.engineUp + 0.4;
  const started = { engine: AT.engine, discord: AT.discord[1], scraper: AT.scraper[1] }[selected];
  const elapsed = started !== undefined && status !== "stopped" ? ` · ${Math.floor(t - started)}s` : "";
  const label = status === "done" ? "exit 0" : status;

  return (
    <div className="flex h-full flex-col bg-[#16161a] px-2 pb-1 pt-1 font-mono text-[15px] leading-[22px] text-[#abb2bf]">
      <p className="whitespace-pre">
        <span className="bg-[#56b6c2] font-bold text-black"> sparky </span>
        <span className="bg-white text-black">{command ? " COMMAND " : searching ? " SEARCH " : " NORMAL "}</span>
        <span style={{ color: engineUp ? C.green : C.red }}>{` ${engineUp ? "●" : "○"} engine`}</span>
        <span style={{ color: C.green }}> ● model</span>
        <span style={{ color: C.green }}> ● phoenix</span>
      </p>
      <div className="mt-3 flex min-h-0 flex-1 gap-0">
        <div className="w-[34ch] shrink-0">
          {pane({ title: " units ", focused: true, children: (<>
            {GROUPS.map(([group, units]) => (
              <div key={group}>
                <p className="whitespace-pre font-bold" style={{ color: C.dim }}>{` ${group}`}</p>
                {units.map((u) => {
                  const s = u.status?.(t) ?? "stopped";
                  const [glyph, color] = GLYPH[s];
                  const port = u.port ?? "";
                  const width = 24 - port.length;
                  const name = u.id.length > width ? `${u.id.slice(0, width - 1)}…` : u.id.padEnd(width);
                  return (
                    <p key={u.id} className={`whitespace-pre ${u.id === selected ? "bg-[#282c34] font-bold text-white" : ""}`}>
                      <span style={{ color }}>{`  ${glyph} `}</span>
                      {name}
                      <span style={{ color: C.dim }}>{port}</span>
                    </p>
                  );
                })}
              </div>
            ))}
          </>) })}
        </div>
        <div className="min-w-0 flex-1">
          {pane({ title: ` ${selected} · ${label}${elapsed} · ${logs.length} lines · follow `, focused: false, footer: unit.hint, children: (<>
            {logs.length === 0 ? (
              <p className="whitespace-pre" style={{ color: C.dim }}>
                {"  nothing captured yet — press enter to start"}
              </p>
            ) : (
              logs.map((l, i) => {
                const match = search !== "" && t >= AT.hit && l.text.toLowerCase().includes(search);
                return (
                  <p key={l.text} className="whitespace-pre">
                    <span style={{ color: C.dim }}>{`${clock(l.at)} `}</span>
                    <span
                      className={`${l.meta ? "italic" : ""} ${match && i === hit ? "bg-[#e5c07b] font-bold text-[#16161a]" : ""}`}
                      style={match && i !== hit ? { color: C.yellow } : l.meta ? { color: C.accent } : undefined}
                    >
                      {l.text}
                    </span>
                  </p>
                );
              })
            )}
          </>) })}
        </div>
      </div>
      <p className="mt-3 whitespace-pre">
        {command ? (
          <>
            <span style={{ color: C.accent }}>:</span>
            {typed(TASK, t, AT.command[0], AT.command[1])}
            <span style={{ color: C.accent }}>█</span>
          </>
        ) : searching ? (
          <>
            <span style={{ color: C.yellow }}>/</span>
            {search}
            <span style={{ color: C.yellow }}>█</span>
          </>
        ) : (
          <>
            {HINTS.map(([k, w]) => (
              <span key={k}>
                <span className="font-bold" style={{ color: C.accent }}>{` ${k}`}</span>
                <span style={{ color: C.dim }}>{` ${w}`}</span>
              </span>
            ))}
            <span style={{ color: C.dim }}>{"   engine http://localhost:8080 · phoenix http://localhost:6006"}</span>
          </>
        )}
      </p>
    </div>
  );
};

/** The shell before the console takes over the terminal. */
const shellAt = (t: number) => (
  <div className="h-full bg-[#16161a] p-6 font-mono text-[19px] leading-7 text-[#c9d1d9]">
    <p>
      <Prompt dir="~/SparkyAI" />
      {typed("just cli", t, AT.shell[0], AT.shell[1])}
      <Caret t={t} />
    </p>
  </div>
);

export const sparkyConsole: Scene = {
  title: "Run it all from the sparky console",
  length: 13.6,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.3, x: 640, y: 300, s: 1.35 },
    { t: 1.2, x: 640, y: 300, s: 1.35 },
    { t: 1.8, x: 800, y: 500, s: 1 },
    { t: 2.3, x: 560, y: 760, s: 1.45 },
    { t: 3.6, x: 560, y: 760, s: 1.45 },
    { t: 4.1, x: 760, y: 420, s: 1.2 },
    { t: 6.4, x: 760, y: 440, s: 1.2 },
    { t: 7.0, x: 900, y: 250, s: 1.4 },
    { t: 9.6, x: 900, y: 250, s: 1.4 },
    { t: 10.1, x: 760, y: 440, s: 1.15 },
    { t: 13.0, x: 800, y: 500, s: 1 },
  ],
  pointer: [{ t: 0, x: 1180, y: 700, s: 1 }],
  clicks: [],
  keys: [
    { keys: ["↵"], t: AT.shell[1] + 0.1 },
    { keys: [":"], t: AT.colon },
    { keys: ["↵"], t: AT.run - 0.1 },
    { keys: ["k"], t: AT.climb },
    { keys: ["↵"], t: AT.engine - 0.1 },
    { keys: ["/"], t: AT.slash },
    { keys: ["j"], t: AT.discord[0] },
    { keys: ["↵"], t: AT.discord[1] },
    { keys: ["j"], t: AT.scraper[0] },
    { keys: ["↵"], t: AT.scraper[1] },
  ],
  view: (t) => <MacWindow title="~/SparkyAI — just cli">{t < AT.tui ? shellAt(t) : consoleAt(t)}</MacWindow>,
};
