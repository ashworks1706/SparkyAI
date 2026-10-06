import type { ReactNode } from "react";
import { finishedCard, runningCard, storedResult, thinkingStep, thoughtStep, threadName, toolDone, toolStarted } from "@/components/discord/format";
import Message from "@/components/discord/Message";
import { SPARKY, type Author } from "@/components/discord/people";
import Window, { Composer, type Server } from "@/components/discord/Window";
import { Keys, Pointer, Stage, ThreadChip } from "./Stage";
import { at, pressAt, progress, typed, type Shot } from "./timeline";

/** Length of the setup video, in seconds. */
export const LENGTH = 30;

/** Where every window sits on the stage. */
const FRAME = { x: 80, y: 70, w: 1440, h: 860 };

/** The campus in the video. example.edu is reserved, so it names no real university. */
const CAMPUS = {
  name: "Example University",
  host: "registrar.example.edu",
  page: "https://registrar.example.edu/calendar",
};

/** When each scene is on screen, in seconds. */
const SCENES = {
  clone: [0, 5.9],
  browse: [5.9, 9.7],
  edit: [9.7, 17.0],
  deploy: [17.0, 21.2],
  discord: [21.2, LENGTH + 1],
} as const;

/** Camera stops. */
const SHOTS: Shot[] = [
  { t: 0, x: 800, y: 500, s: 1 },
  { t: 0.4, x: 800, y: 500, s: 1 },
  { t: 1.1, x: 640, y: 300, s: 1.35 },
  { t: 3.4, x: 640, y: 380, s: 1.35 },
  { t: 5.4, x: 700, y: 500, s: 1.2 },
  { t: 6.2, x: 800, y: 500, s: 1 },
  { t: 6.9, x: 520, y: 150, s: 1.9 },
  { t: 8.8, x: 520, y: 150, s: 1.9 },
  { t: 9.5, x: 800, y: 500, s: 1 },
  { t: 10.2, x: 880, y: 420, s: 1.3 },
  { t: 12.8, x: 880, y: 420, s: 1.3 },
  { t: 13.4, x: 880, y: 420, s: 1.3 },
  { t: 14.8, x: 880, y: 420, s: 1.3 },
  { t: 15.7, x: 800, y: 360, s: 1.5 },
  { t: 16.6, x: 800, y: 360, s: 1.5 },
  { t: 17.2, x: 800, y: 500, s: 1 },
  { t: 17.8, x: 640, y: 260, s: 1.45 },
  { t: 20.6, x: 640, y: 300, s: 1.45 },
  { t: 21.4, x: 800, y: 500, s: 1 },
  { t: 22.4, x: 800, y: 500, s: 1 },
  { t: 23.2, x: 1290, y: 600, s: 1.5 },
  { t: 27.8, x: 1290, y: 640, s: 1.55 },
  { t: 29.0, x: 800, y: 500, s: 1 },
];

/** Pointer stops. */
const POINTER: Shot[] = [
  { t: 0, x: 1180, y: 700, s: 1 },
  { t: 6.4, x: 1000, y: 560, s: 1 },
  { t: 7.3, x: 560, y: 152, s: 1 },
  { t: 9.6, x: 560, y: 152, s: 1 },
  { t: 10.4, x: 1200, y: 600, s: 1 },
  { t: 12.9, x: 1200, y: 600, s: 1 },
  { t: 13.2, x: 580, y: 136, s: 1 },
  { t: 15.3, x: 580, y: 136, s: 1 },
  { t: 15.6, x: 742, y: 136, s: 1 },
  { t: 17.0, x: 1180, y: 700, s: 1 },
  { t: 22.0, x: 1180, y: 700, s: 1 },
  { t: 23.6, x: 1400, y: 470, s: 1 },
  { t: 26.6, x: 1400, y: 470, s: 1 },
  { t: 27.5, x: 1200, y: 806, s: 1 },
];

/** Pointer presses. */
const CLICKS = [7.4, 13.25, 15.65];

/** Keystrokes shown at the bottom: the keys, and when they show. */
const KEYS: { keys: string[]; t: number }[] = [
  { keys: ["⌘", "C"], t: 7.8 },
  { keys: ["⌘", "V"], t: 11.2 },
  { keys: ["⌘", "S"], t: 16.55 },
  { keys: ["↵"], t: 19.8 },
];

/** One line in a terminal: a command typed between at and to, or output printed at at. */
type Line = { at: number; to?: number; text: string; tone?: "ok" | "dim" };

/** The first terminal: clone the repository and bootstrap it. Output is what git and the justfile print. */
const CLONE: Line[] = [
  { at: 0.5, to: 2.0, text: "git clone https://github.com/ashworks1706/SparkyAI && cd SparkyAI" },
  { at: 2.2, text: "Cloning into 'SparkyAI'...", tone: "dim" },
  { at: 2.5, text: "Receiving objects: 100% (9418/9418), 6.12 MiB | 18.4 MiB/s, done.", tone: "dim" },
  { at: 2.8, text: "Resolving deltas: 100% (5873/5873), done.", tone: "dim" },
  { at: 3.1, to: 3.7, text: "just bootstrap" },
  { at: 3.9, text: "created .env, fill in tokens and model URLs" },
  { at: 4.1, text: "hooks installed: .githooks/pre-commit" },
  { at: 4.4, text: " ✔ Container deploy-postgres-1  Healthy", tone: "ok" },
  { at: 4.5, text: " ✔ Container deploy-redis-1     Healthy", tone: "ok" },
  { at: 4.6, text: " ✔ Container deploy-minio-1     Healthy", tone: "ok" },
  { at: 4.8, text: "applied: 0001_init, 0002_query_sources, 0003_summary_turns, …" },
  { at: 5.0, text: "ready. Next a model, because the engine answers nothing without one:" },
  { at: 5.1, text: "  GPU:    just model       llama-server on CUDA", tone: "dim" },
  { at: 5.2, text: "  no GPU: just model-cpu   the same models on the processor, slowly", tone: "dim" },
  { at: 5.3, text: "  hosted: set SPARKY_MODEL__BASE_URL and SPARKY_MODEL__API_KEY in .env", tone: "dim" },
];

/** The second terminal: index the new page and start the stack. */
const DEPLOY: Line[] = [
  { at: 17.4, to: 18.5, text: "just scraper run --category calendar" },
  { at: 18.9, text: "academic_calendar: indexed (38 chunks)", tone: "ok" },
  { at: 19.2, to: 19.7, text: "just up" },
  { at: 20.0, text: " ✔ Container deploy-postgres-1  Running", tone: "ok" },
  { at: 20.1, text: " ✔ Container deploy-scraper-1   Started", tone: "ok" },
  { at: 20.2, text: " ✔ Container deploy-engine-1    Started", tone: "ok" },
  { at: 20.3, text: " ✔ Container deploy-discord-1   Started", tone: "ok" },
];

/** A macOS window: traffic lights, a title, and its body. */
const MacWindow = ({ title, dark = true, children }: { title: ReactNode; dark?: boolean; children: ReactNode }) => (
  <div
    className={`flex h-full w-full flex-col overflow-hidden rounded-xl shadow-[0_40px_80px_-30px_rgba(0,0,0,0.6)] ring-1 ${dark ? "bg-[#16161a] ring-black/50" : "bg-white ring-black/20"}`}
  >
    <div className={`relative flex h-11 shrink-0 items-center px-4 ${dark ? "bg-[#26262b] text-[#9c9ca6]" : "bg-[#ececee] text-[#55555c]"}`}>
      <span className="flex gap-2">
        <span className="h-3.5 w-3.5 rounded-full bg-[#ff5f57]" />
        <span className="h-3.5 w-3.5 rounded-full bg-[#febc2e]" />
        <span className="h-3.5 w-3.5 rounded-full bg-[#28c840]" />
      </span>
      <div className="absolute inset-x-0 flex justify-center text-sm font-medium">{title}</div>
    </div>
    <div className="min-h-0 flex-1">{children}</div>
  </div>
);

/** A shell prompt, as zsh shows it in the repository. */
const Prompt = ({ dir }: { dir: string }) => (
  <>
    <span className="text-[#7ee787]">dev@laptop</span> <span className="text-[#79c0ff]">{dir}</span> <span className="text-[#c9d1d9]">$</span>{" "}
  </>
);

/** A terminal that plays lines in time, with a caret on the line being typed. */
const Terminal = ({ lines, t, dir, after }: { lines: Line[]; t: number; dir: (i: number) => string; after: string }) => {
  const shown = lines.filter((l) => l.at <= t);
  const caret = Math.floor(t * 2.2) % 2 === 0;
  const typing = shown.findIndex((l) => l.to !== undefined && t < l.to);
  const last = shown[shown.length - 1];
  const idle = typing === -1 && (!last || last.to === undefined || t >= last.to);
  return (
    <MacWindow title={`${after} — zsh`}>
      <div className="space-y-1.5 p-6 font-mono text-[19px] leading-7 text-[#c9d1d9]">
        {shown.map((l, i) =>
          l.to !== undefined ? (
            <p key={l.text}>
              <Prompt dir={dir(i)} />
              {typed(l.text, t, l.at, l.to)}
              {i === typing && <span className={`inline-block h-6 w-2.5 translate-y-1 bg-[#c9d1d9] ${caret ? "" : "opacity-0"}`} />}
            </p>
          ) : (
            <p key={l.text} className={l.tone === "ok" ? "text-[#7ee787]" : l.tone === "dim" ? "text-[#8b949e]" : ""}>
              {l.text}
            </p>
          ),
        )}
        {idle && (
          <p>
            <Prompt dir="~/SparkyAI" />
            <span className={`inline-block h-6 w-2.5 translate-y-1 bg-[#c9d1d9] ${caret ? "" : "opacity-0"}`} />
          </p>
        )}
      </div>
    </MacWindow>
  );
};

/** The registrar page the developer takes a URL from. */
const Browser = ({ selected }: { selected: boolean }) => (
  <MacWindow
    dark={false}
    title={
      <span className="mt-0.5 flex h-8 w-[560px] items-center gap-2 rounded-lg bg-white px-4 text-[15px] text-[#3c3c43] ring-1 ring-black/10">
        <svg aria-hidden viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
          <path d="M7 10V7a5 5 0 0 1 10 0v3h1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V11a1 1 0 0 1 1-1h1Zm2 0h6V7a3 3 0 0 0-6 0v3Z" />
        </svg>
        <span className={selected ? "rounded-sm bg-[#b3d7ff]" : ""}>{CAMPUS.page.replace("https://", "")}</span>
      </span>
    }
  >
    <div className="h-full bg-white font-sans text-[#1b1f2a]">
      <div className="flex h-16 items-center gap-4 bg-[#1d3557] px-10 text-white">
        <span className="font-serif text-xl font-bold">{CAMPUS.name}</span>
        <span className="text-white/60">|</span>
        <span className="text-white/80">Office of the Registrar</span>
      </div>
      <div className="mx-auto max-w-4xl px-10 py-10">
        <h1 className="font-serif text-4xl font-bold">Academic Calendar</h1>
        <p className="mt-2 text-[#5b6170]">Spring 2027 dates and deadlines.</p>
        <table className="mt-8 w-full text-left text-lg">
          <tbody className="divide-y divide-[#e3e6ec]">
            {[
              ["Classes begin", "January 11"],
              ["Last day to add or drop", "January 19"],
              ["Spring break", "March 8 to 14"],
              ["Last day of classes", "April 30"],
              ["Final exams", "May 3 to 8"],
            ].map(([what, when]) => (
              <tr key={what}>
                <td className="py-3.5">{what}</td>
                <td className="py-3.5 font-semibold">{when}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </MacWindow>
);

/** One line of code: its text split into coloured runs. */
type Code = { text: string; tone?: "key" | "str" | "num" | "com" | "kw" }[];

const TONES = { key: "text-[#9cdcfe]", str: "text-[#ce9178]", num: "text-[#b5cea8]", com: "text-[#6a9955]", kw: "text-[#c586c0]" };

/** The new page line in pages.py: typed, then the URL pasted, then the rest typed. */
const pageLine = (t: number): Code => {
  const head = typed('_page("academic_calendar", "', t, 10.3, 11.1);
  const url = t >= 11.25 ? CAMPUS.page : "";
  const tail = t >= 11.25 ? typed('", "calendar", 72),', t, 11.5, 12.4) : "";
  return [{ text: "    " }, { text: head, tone: head.length > 6 ? "str" : undefined }, { text: url, tone: "str" }, { text: tail, tone: "str" }];
};

/** The files open in the editor, and which lines each shows at time t. */
const FILES = {
  pages: {
    name: "pages.py",
    path: "apps/scraper/sources",
    lines: (t: number): Code[] => [
      [{ text: '"""Static campus pages: one source each, indexed from the markdown Firecrawl returns."""', tone: "com" }],
      [],
      [{ text: "from", tone: "kw" }, { text: " scraper.core.types " }, { text: "import", tone: "kw" }, { text: " Source" }],
      [],
      [{ text: "PAGES: tuple[Source, ...] = (" }],
      [{ text: "    # calendar", tone: "com" }],
      pageLine(t),
      [{ text: "    # registrar", tone: "com" }],
      [{ text: "    _page(" }, { text: '"registrar_drop_add", "https://registrar.example.edu/drop-add", "registrar"', tone: "str" }, { text: ")," }],
      [{ text: "    _page(" }, { text: '"registrar_grades", "https://registrar.example.edu/grades", "registrar"', tone: "str" }, { text: ")," }],
      [{ text: ")" }],
    ],
    active: 6,
  },
  toml: {
    name: "sparky.toml",
    path: "",
    lines: (t: number): Code[] => [
      [{ text: "[prompt]", tone: "kw" }],
      [
        { text: "utc_offset_hours", tone: "key" },
        { text: " = " },
        { text: t >= 13.6 ? "-5" : "-7", tone: "num" },
        { text: t >= 13.6 ? `          # ${typed("Eastern, standard time", t, 13.7, 14.5)}` : "          # Arizona, no daylight saving", tone: "com" },
      ],
      [],
      [{ text: "[tools]", tone: "kw" }],
      [{ text: "search", tone: "key" }, { text: " = " }, { text: "true", tone: "num" }],
      [
        { text: "knowledge_description", tone: "key" },
        { text: " = " },
        { text: `"Search the stored knowledge base of ${t >= 14.8 ? typed(CAMPUS.name, t, 14.8, 15.3) : "ASU"} pages: programs, policies…"`, tone: "str" },
      ],
    ],
    active: -1,
  },
  env: {
    name: ".env",
    path: "",
    lines: (t: number): Code[] => [
      [{ text: "SPARKY_DISCORD__TOKEN", tone: "key" }, { text: "=" }, { text: typed("•".repeat(28), t, 15.85, 16.25), tone: "str" }],
      [{ text: "SPARKY_DISCORD__GUILD_ID", tone: "key" }, { text: "=" }, { text: t >= 16.35 ? "1290457783312334848" : "0", tone: "num" }],
      [{ text: "SPARKY_MODEL__BASE_URL", tone: "key" }, { text: "=" }, { text: "http://localhost:8000/v1", tone: "str" }],
    ],
    active: -1,
  },
};

/** Which editor tab is open at time t. */
const fileAt = (t: number) => (t < 13.25 ? FILES.pages : t < 15.65 ? FILES.toml : FILES.env);

/** A code editor with the explorer, tabs, and the open file. */
const Editor = ({ t }: { t: number }) => {
  const file = fileAt(t);
  const tabs = [FILES.pages, FILES.toml, FILES.env];
  return (
    <MacWindow title="SparkyAI">
      <div className="flex h-full bg-[#1e1e1e] font-mono text-[17px] text-[#d4d4d4]">
        <aside className="w-64 shrink-0 bg-[#181818] px-3 py-4 font-sans text-[15px] text-[#cccccc]">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[#8b8b8b]">Explorer</p>
          {["apps/", "  scraper/", "    sources/", "      pages.py", "deploy/", ".env", "justfile", "sparky.toml"].map((row) => (
            <p
              key={row}
              className={`whitespace-pre rounded px-2 py-0.5 ${row.trim() === file.name ? "bg-[#37373d] text-white" : ""}`}
            >
              {row}
            </p>
          ))}
        </aside>
        <div className="min-w-0 flex-1">
          <div className="flex h-10 bg-[#181818] font-sans text-[15px]">
            {tabs.map((tab) => (
              <span
                key={tab.name}
                className={`flex w-[162px] items-center border-r border-[#2b2b2b] px-4 ${tab === file ? "border-t-2 border-t-[#0078d4] bg-[#1e1e1e] text-white" : "text-[#9d9d9d]"}`}
              >
                {tab.name}
              </span>
            ))}
          </div>
          <div className="py-3">
            {file.lines(t).map((line, i) => (
              <p key={i} className={`flex whitespace-pre ${i === file.active && t < 12.9 ? "bg-[#2a2d2e]" : ""}`}>
                <span className="w-14 shrink-0 pr-4 text-right text-[#6e7681]">{i + 1}</span>
                {line.map((run, k) => (
                  <span key={k} className={run.tone ? TONES[run.tone] : ""}>
                    {run.text}
                  </span>
                ))}
              </p>
            ))}
          </div>
        </div>
      </div>
    </MacWindow>
  );
};

/** The campus Discord the bot answers in once deployed. */
const SERVER: Server = { name: CAMPUS.name, badge: "EU", className: "bg-[#1d3557] text-white" };

/** The first student to ask the newly deployed bot. */
const ASKER: Author = { name: "jordan", color: "#f0b232" };

const QUESTION = "when does spring break start?";
const SEARCH = { query: "spring break 2027 dates" };
const RESULT = storedResult(
  "Academic Calendar",
  CAMPUS.page,
  "2 hours ago",
  "Spring 2027. Classes begin January 11. Last day to add or drop January 19. Spring break March 8 to 14.",
);
const THOUGHT = thoughtStep("Spring break dates are on the academic calendar; search the stored pages for spring 2027.");
const WRAP = thoughtStep("The calendar lists the dates; answer with them and the page.");
const ANSWER = "Spring break runs **March 8 to 14**. Classes start again on Monday, March 15.";

/** Sparky's card at time t, edits 1.5 s apart. */
const card = (t: number) => {
  const done = toolDone("search_knowledge", SEARCH, RESULT);
  if (t >= 26.6) return finishedCard([THOUGHT, done, WRAP], ANSWER);
  if (t >= 25.1) return runningCard([THOUGHT, done, thinkingStep()], 2);
  if (t >= 23.6) return runningCard([THOUGHT, toolStarted("search_knowledge", SEARCH)], 1);
  if (t >= 22.1) return runningCard([thinkingStep()], 0);
  return null;
};

/** The deployed bot answering its first question in the campus server. */
const Discord = ({ t }: { t: number }) => {
  const reply = card(t);
  const name = threadName(QUESTION);
  return (
    <Window
      server={SERVER}
      channel="ask-sparky"
      topic={`Ask Sparky about ${CAMPUS.name}. Mention it to start a thread.`}
      me={ASKER}
      messages={
        <Message author={ASKER} time="Today at 9:14 AM" text={`@Sparky ${QUESTION}`}>
          <ThreadChip name={name} count={reply ? "1 Message" : "See Thread"} />
        </Message>
      }
      composer={<Composer placeholder="Message #ask-sparky" />}
      thread={{
        name,
        width: 460,
        body: (
          <div style={{ width: 460 }}>
            <Message author={ASKER} time="Today at 9:14 AM" text={`@Sparky ${QUESTION}`} />
            {reply && (
              <Message
                author={SPARKY}
                time="Today at 9:14 AM"
                text={reply}
                buttons={t >= 26.6 ? [{ label: "Academic Calendar", href: CAMPUS.page }] : []}
                hovered={t >= 27.5 ? 0 : undefined}
              />
            )}
          </div>
        ),
      }}
    />
  );
};

/** How visible a scene is at time t: fades in at its start and out at its end. */
const shown = (t: number, [from, to]: readonly [number, number]) =>
  Math.min(from === 0 ? 1 : progress(t, from - 0.15, 0.3), 1 - progress(t, to - 0.15, 0.3));

/** One frame of the setup video at time t. */
export const Frame = ({ t }: { t: number }) => {
  const shot = at(SHOTS, t);
  const pointer = at(POINTER, t);
  const key = KEYS.find((k) => t >= k.t && t < k.t + 0.9);
  const keyOpacity = key ? Math.min(progress(t, key.t, 0.12), 1 - progress(t, key.t + 0.7, 0.2)) : 0;
  const scenes: [readonly [number, number], ReactNode][] = [
    [SCENES.clone, <Terminal lines={CLONE} t={t} dir={(i) => (i === 0 ? "~" : "~/SparkyAI")} after="~/SparkyAI" />],
    [SCENES.browse, <Browser selected={t >= 7.45} />],
    [SCENES.edit, <Editor t={t} />],
    [SCENES.deploy, <Terminal lines={DEPLOY} t={t} dir={() => "~/SparkyAI"} after="~/SparkyAI" />],
    [SCENES.discord, <Discord t={t} />],
  ];
  return (
    <Stage shot={shot} overlay={key && <Keys keys={key.keys} opacity={keyOpacity} />}>
      {scenes.map(([span, scene], i) => {
        const opacity = shown(t, span);
        return opacity > 0 ? (
          <div
            key={i}
            className="absolute"
            style={{ left: FRAME.x, top: FRAME.y, width: FRAME.w, height: FRAME.h, opacity, transform: `scale(${0.97 + 0.03 * opacity})` }}
          >
            {scene}
          </div>
        ) : null;
      })}
      <Pointer x={pointer.x} y={pointer.y} pressed={pressAt(CLICKS, t)} />
    </Stage>
  );
};
