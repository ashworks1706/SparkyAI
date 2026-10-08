import { Caret } from "../../Caret";
import { typed, type Scene } from "../../timeline";
import { AT, COLORS, GROUPS, HINTS, LOGS, TASK, type Status } from "../console-data";
import { MacWindow, Prompt } from "../kit";
import { Pane } from "../Pane";

/** Seconds per keypress while the selection climbs to the engine. */
const STEP = 0.07;

/** The glyph and colour each unit status shows in the unit list. */
const GLYPH: Record<Status, [string, string]> = {
  stopped: ["○", COLORS.dim],
  starting: ["◐", COLORS.yellow],
  running: ["●", COLORS.green],
  done: ["✓", COLORS.blue],
};

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

/** The wall clock the log pane prints, counting from the scene start. */
const clock = (at: number) => `09:02:${String(10 + Math.floor(at)).padStart(2, "0")}`;

/**
 * The sparky developer console at scene time t, drawn as apps/cli/src/app/ui.rs draws it:
 * status bar, unit list by group, the selected unit's log pane, and the command line.
 */
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
        <span style={{ color: engineUp ? COLORS.green : COLORS.red }}>{` ${engineUp ? "●" : "○"} engine`}</span>
        <span style={{ color: COLORS.green }}> ● model</span>
        <span style={{ color: COLORS.green }}> ● phoenix</span>
      </p>
      <div className="mt-3 flex min-h-0 flex-1 gap-0">
        <div className="w-[34ch] shrink-0">
          <Pane title=" units " focused>
            {GROUPS.map(([group, units]) => (
              <div key={group}>
                <p className="whitespace-pre font-bold" style={{ color: COLORS.dim }}>{` ${group}`}</p>
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
                      <span style={{ color: COLORS.dim }}>{port}</span>
                    </p>
                  );
                })}
              </div>
            ))}
          </Pane>
        </div>
        <div className="min-w-0 flex-1">
          <Pane title={` ${selected} · ${label}${elapsed} · ${logs.length} lines · follow `} focused={false} footer={unit.hint}>
            {logs.length === 0 ? (
              <p className="whitespace-pre" style={{ color: COLORS.dim }}>
                {"  nothing captured yet — press enter to start"}
              </p>
            ) : (
              logs.map((l, i) => {
                const match = search !== "" && t >= AT.hit && l.text.toLowerCase().includes(search);
                return (
                  <p key={l.text} className="whitespace-pre">
                    <span style={{ color: COLORS.dim }}>{`${clock(l.at)} `}</span>
                    <span
                      className={`${l.meta ? "italic" : ""} ${match && i === hit ? "bg-[#e5c07b] font-bold text-[#16161a]" : ""}`}
                      style={match && i !== hit ? { color: COLORS.yellow } : l.meta ? { color: COLORS.accent } : undefined}
                    >
                      {l.text}
                    </span>
                  </p>
                );
              })
            )}
          </Pane>
        </div>
      </div>
      <p className="mt-3 whitespace-pre">
        {command ? (
          <>
            <span style={{ color: COLORS.accent }}>:</span>
            {typed(TASK, t, AT.command[0], AT.command[1])}
            <span style={{ color: COLORS.accent }}>█</span>
          </>
        ) : searching ? (
          <>
            <span style={{ color: COLORS.yellow }}>/</span>
            {search}
            <span style={{ color: COLORS.yellow }}>█</span>
          </>
        ) : (
          <>
            {HINTS.map(([k, w]) => (
              <span key={k}>
                <span className="font-bold" style={{ color: COLORS.accent }}>{` ${k}`}</span>
                <span style={{ color: COLORS.dim }}>{` ${w}`}</span>
              </span>
            ))}
            <span style={{ color: COLORS.dim }}>{"   engine http://localhost:8080 · phoenix http://localhost:6006"}</span>
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

/** Scene: start the console, run a scraper task, then start the engine, the bot and the scraper. */
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
