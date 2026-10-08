import { typed, type Scene } from "../timeline";
import { CAMPUS, MCP } from "./campus";
import { Editor, MacWindow, Terminal, type Line } from "./kit";
import { charCount, code, firstChars, type Code } from "./text";

/** The repository tree the explorer shows. */
const EXPLORER = [
  "apps/",
  "  engine/src/runtime/tools/",
  "    knowledge/search/",
  "  scraper/",
  "    query/",
  "      sources/",
  "    sources/",
  "      pages.py",
  "deploy/",
  ".env",
  "justfile",
  "sparky.toml",
];

/** The explorer with a new file listed under its folder. */
const withFile = (rows: string[], after: string, file: string) => {
  const i = rows.indexOf(after);
  return [...rows.slice(0, i + 1), file, ...rows.slice(i + 1)];
};

/** Where the window sits, so camera stops can name lines of code. */
const CODE = { x: 420, y: 178, line: 25, char: 9.6 };

/** The stage point of column col on line i of the editor. */
const spot = (i: number, col = 0, s = 1.3) => ({ x: CODE.x + col * CODE.char, y: CODE.y + i * CODE.line, s });

/** Clone the repository and bootstrap it. Output is what git and the justfile print. */
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

const REST = { t: 0, x: 1180, y: 700, s: 1 };

export const clone: Scene = {
  title: "Clone and bootstrap",
  length: 5.8,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.4, x: 800, y: 500, s: 1 },
    { t: 1.1, x: 640, y: 300, s: 1.35 },
    { t: 3.4, x: 640, y: 380, s: 1.35 },
    { t: 5.4, x: 700, y: 500, s: 1.2 },
  ],
  pointer: [REST],
  clicks: [],
  keys: [],
  view: (t) => <Terminal lines={CLONE} t={t} dir={(i) => (i === 0 ? "~" : "~/SparkyAI")} />,
};

/** The registrar page the developer takes a URL from. */
const browserAt = (selected: boolean) => (
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

export const browse: Scene = {
  title: "Pick a page from your campus site",
  length: 3.8,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.3, x: 800, y: 500, s: 1 },
    { t: 1.0, x: 520, y: 150, s: 1.9 },
    { t: 2.9, x: 520, y: 150, s: 1.9 },
    { t: 3.6, x: 800, y: 500, s: 1 },
  ],
  pointer: [
    { t: 0, x: 1180, y: 700, s: 1 },
    { t: 0.5, x: 1000, y: 560, s: 1 },
    { t: 1.4, x: 560, y: 152, s: 1 },
  ],
  clicks: [1.5],
  keys: [{ keys: ["⌘", "C"], t: 1.9 }],
  view: (t) => browserAt(t >= 1.55),
};

/** pages.py with the new page line: typed, the URL pasted, then the rest typed. */
const pagesFile = (t: number): Code[] => {
  const head = typed('_page("academic_calendar", "', t, 0.6, 1.4);
  const url = t >= 1.55 ? CAMPUS.page : "";
  const tail = t >= 1.55 ? typed('", "calendar", 72),', t, 1.8, 2.7) : "";
  return [
    code(['"""Static campus pages: one source each, indexed from the markdown Firecrawl returns."""', "com"]),
    [],
    code(["from", "kw"], " scraper.core.types ", ["import", "kw"], " Source"),
    [],
    code("PAGES: tuple[Source, ...] = ("),
    code(["    # calendar", "com"]),
    code("    ", [head, "str"], [url, "str"], [tail, "str"]),
    code(["    # registrar", "com"]),
    code("    _page(", [`"registrar_drop_add", "https://${CAMPUS.registrar}/drop-add", "registrar"`, "str"], "),"),
    code("    _page(", [`"registrar_grades", "https://${CAMPUS.registrar}/grades", "registrar"`, "str"], "),"),
    code(")"),
  ];
};

export const pages: Scene = {
  title: "Index it as a static page",
  length: 3.6,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.5, x: 860, y: 340, s: 1.3 },
    { t: 3.0, x: 860, y: 340, s: 1.3 },
  ],
  pointer: [REST],
  clicks: [],
  keys: [
    { keys: ["⌘", "V"], t: 1.5 },
    { keys: ["⌘", "S"], t: 2.85 },
  ],
  view: (t) => (
    <Editor explorer={EXPLORER} tabs={["pages.py"]} open="pages.py" lines={pagesFile(t)} cursor={t < 3 ? 6 : undefined} t={t} />
  ),
};

/** The scraper side of a live source: where to fetch and how to read the page. */
const REC_PY: Code[] = [
  code(['"""Rec center and pool hours at Example University, this week."""', "com"]),
  [],
  code(["from", "kw"], " __future__ ", ["import", "kw"], " annotations"),
  [],
  code(["from", "kw"], " scraper.core.types ", ["import", "kw"], " QuerySource"),
  code(["from", "kw"], " scraper.ingest.extract ", ["import", "kw"], " page_text"),
  [],
  [],
  code(["def", "kw"], " ", ["to_url", "fn"], "(params: ", ["dict", "ty"], "[", ["str", "ty"], ", ", ["str", "ty"], "]) -> ", ["str", "ty"], ":"),
  code(['    """The hours page. It takes no parameters."""', "com"]),
  code("    ", ["return", "kw"], " ", [`"${CAMPUS.rec}"`, "str"]),
  [],
  [],
  code("QUERY = ", ["QuerySource", "ty"], "("),
  code("    ", ["key", "key"], "=", ['"rec_hours"', "str"], ","),
  code("    ", ["description", "key"], "=", ['"Fetch this week\'s hours at the rec center and pool."', "str"], ","),
  code("    ", ["params", "key"], "=(),"),
  code("    ", ["to_url", "key"], "=to_url,"),
  code("    ", ["extractor", "key"], "=page_text,"),
  code("    ", ["category", "key"], "=", ['"recreation"', "str"], ","),
  code(")"),
];

/** The engine side: the key, the label its button carries, and its parameters. */
const method = (name: string, ret: string, body: Code): Code[] => [
  code("    ", ["fn", "kw"], " ", [name, "fn"], "(&", ["self", "kw"], ") -> ", [ret, "ty"], " {"),
  [{ text: "        " }, ...body],
  code("    }"),
  [],
];

const REC_RS: Code[] = [
  code(["//! This week's hours at the Example University rec center and pool.", "com"]),
  [],
  code(["use", "kw"], " super::{", ["LiveSource", "ty"], ", ", ["Param", "ty"], "};"),
  [],
  code(["/// The rec center hours page.", "com"]),
  code(["pub struct", "kw"], " ", ["RecHours", "ty"], ";"),
  [],
  code(["impl", "kw"], " ", ["LiveSource", "ty"], " ", ["for", "kw"], " ", ["RecHours", "ty"], " {"),
  ...method("key", "&'static str", code(['"rec_hours"', "str"])),
  ...method("hint", "&'static str", code(['"rec center and pool hours"', "str"])),
  ...method("label", "&'static str", code([`"${CAMPUS.recLabel}"`, "str"])),
  ...method("category", "&'static str", code(['"recreation"', "str"])),
  ...method("params", "&'static [Param]", code("&[]")).slice(0, 3),
  code("}"),
];

/** registry.py with the new module imported and listed. */
const registry = (t: number): { lines: Code[]; added: number[] } => {
  const imported = t >= 0.3;
  const listed = t >= 0.6;
  const lines = [
    code(["from", "kw"], " scraper.query.sources ", ["import", "kw"], " ("),
    code("    library_hours,"),
    code("    news,"),
    ...(imported ? [code("    rec_hours,")] : []),
    code("    scholarships,"),
    code("    ..."),
    code(")"),
    [],
    code("_MODULES = ("),
    code("    dining,"),
    code("    jobs,"),
    ...(listed ? [code("    rec_hours,")] : []),
    code("    web,"),
    code(")"),
  ];
  return { lines, added: [imported ? 3 : -1, listed ? 11 : -1] };
};

/** search/mod.rs with the module declared and the source in the catalog. */
const catalog = (t: number): { lines: Code[]; added: number[] } => {
  const declared = t >= 0.3;
  const listed = t >= 0.6;
  const lines = [
    code(["pub mod", "kw"], " news;"),
    ...(declared ? [code(["pub mod", "kw"], " rec_hours;")] : []),
    code(["pub mod", "kw"], " scholarships;"),
    [],
    code(["/// Every live source the engine offers, in the order the model sees them.", "com"]),
    code(["pub fn", "kw"], " ", ["catalog", "fn"], "() -> ", ["Vec", "ty"], "<", ["Box", "ty"], "<", ["dyn", "kw"], " ", ["LiveSource", "ty"], ">> {"),
    code("    ", ["vec!", "fn"], "["),
    code("        ", ["Box", "ty"], "::new(jobs::", ["Jobs", "ty"], "),"),
    ...(listed ? [code("        ", ["Box", "ty"], "::new(rec_hours::", ["RecHours", "ty"], "),")] : []),
    code("        ", ["Box", "ty"], "::new(web::", ["Web", "ty"], "),"),
    code("    ]"),
    code("}"),
  ];
  return { lines, added: [declared ? 1 : -1, listed ? 8 : -1] };
};

/** When each file of the live source is on screen. */
const STEPS = { py: 0, registry: 2.7, rs: 4.1, catalog: 6.8 };

export const source: Scene = {
  title: "Write a live source of your own",
  length: 8.8,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.5, ...spot(10, 34, 1.2) },
    { t: 2.4, ...spot(10, 34, 1.2) },
    { t: 2.9, ...spot(6, 20, 1.6) },
    { t: 3.9, ...spot(6, 20, 1.6) },
    { t: 4.4, ...spot(9, 34, 1.2) },
    { t: 5.6, ...spot(17, 34, 1.2) },
    { t: 6.6, ...spot(17, 34, 1.2) },
    { t: 7.0, ...spot(5, 28, 1.5) },
    { t: 8.4, ...spot(5, 28, 1.5) },
  ],
  pointer: [REST],
  clicks: [],
  keys: [
    { keys: ["⌘", "S"], t: 2.3 },
    { keys: ["⌘", "S"], t: 6.4 },
  ],
  view: (t) => {
    const withPy = withFile(EXPLORER, "      sources/", "        rec_hours.py");
    const rows = t >= STEPS.rs ? withFile(withPy, "    knowledge/search/", "      rec_hours.rs") : withPy;
    if (t < STEPS.registry) {
      const lines = firstChars(REC_PY, Math.floor(charCount(REC_PY) * Math.min(1, Math.max(0, (t - 0.3) / 1.9))));
      return <Editor explorer={rows} tabs={["pages.py", "rec_hours.py"]} open="rec_hours.py" lines={lines} cursor={lines.length - 1} t={t} />;
    }
    if (t < STEPS.rs) {
      const { lines, added } = registry(t - STEPS.registry);
      return <Editor explorer={rows} tabs={["pages.py", "rec_hours.py", "registry.py"]} open="registry.py" lines={lines} added={added} t={t} />;
    }
    if (t < STEPS.catalog) {
      const local = t - STEPS.rs;
      const lines = firstChars(REC_RS, Math.floor(charCount(REC_RS) * Math.min(1, Math.max(0, (local - 0.2) / 2.0))));
      return (
        <Editor explorer={rows} tabs={["rec_hours.py", "registry.py", "rec_hours.rs"]} open="rec_hours.rs" lines={lines} cursor={lines.length - 1} t={t} />
      );
    }
    const { lines, added } = catalog(t - STEPS.catalog);
    return <Editor explorer={rows} tabs={["rec_hours.py", "registry.py", "rec_hours.rs", "mod.rs"]} open="mod.rs" lines={lines} added={added} t={t} />;
  },
};

/** sparky.toml: the time zone, the campus name, Canvas on, and an MCP server. */
const tomlFile = (t: number): { lines: Code[]; added: number[] } => {
  const server = [
    code(["[[mcp.servers]]", "kw"]),
    code(["name", "key"], " = ", [`"${MCP.name}"`, "str"]),
    code(["url", "key"], " = ", [`"${MCP.url}"`, "str"]),
    code(["tools", "key"], " = [", MCP.tools.map((n) => `"${n}"`).join(", "), "]"),
    code(["risks", "key"], " = { ", MCP.tools.map((n) => `${n} = "read_public"`).join(", "), " }"),
  ];
  const typedServer = t >= 5.6 ? firstChars(server, Math.floor(charCount(server) * Math.min(1, (t - 5.6) / 3.0))) : [];
  const lines = [
    code(["[prompt]", "kw"]),
    code(
      ["utc_offset_hours", "key"],
      " = ",
      [t >= 0.9 ? "-5" : "-7", "num"],
      [t >= 0.9 ? `          # ${typed("Eastern, standard time", t, 1.0, 1.8)}` : "          # Arizona, no daylight saving", "com"],
    ),
    [],
    code(["[tools]", "kw"]),
    code(["disabled", "key"], " = []"),
    code(["search", "key"], " = ", ["true", "num"]),
    code(["knowledge_description", "key"], " = ", [`"Search the stored knowledge base of ${t >= 2.1 ? typed(CAMPUS.name, t, 2.1, 2.6) : "ASU"} pages: programs, policies…"`, "str"]),
    [],
    code(["[sandbox]", "kw"]),
    code(["enabled", "key"], " = ", ["true", "num"]),
    code(["egress", "key"], " = ", ["true", "num"]),
    [],
    code(["[canvas]", "kw"]),
    code(["enabled", "key"], " = ", [t >= 3.6 ? "true" : "false", "num"]),
    code(["base_url", "key"], " = ", [`"https://${t >= 4.0 ? typed(CAMPUS.canvas, t, 4.0, 4.6) : "canvas.asu.edu"}"`, "str"]),
    [],
    code(["[papers]", "kw"]),
    code(["enabled", "key"], " = ", ["true", "num"]),
    [],
    code(["[wikipedia]", "kw"]),
    code(["enabled", "key"], " = ", ["true", "num"]),
    [],
    code(["[mcp]", "kw"]),
    code(["required_props_only", "key"], " = ", ["true", "num"]),
    [],
    ...typedServer,
  ];
  const added = [t >= 0.9 ? 1 : -1, t >= 2.1 ? 6 : -1, t >= 3.6 ? 13 : -1, t >= 4.0 ? 14 : -1, ...typedServer.map((_, i) => 25 + i)];
  return { lines, added };
};

export const toml: Scene = {
  title: "Turn on tools and add an MCP server",
  length: 10.4,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.5, ...spot(4, 34, 1.35) },
    { t: 2.8, ...spot(5, 34, 1.35) },
    { t: 3.3, ...spot(13, 24, 1.45) },
    { t: 4.9, ...spot(13, 24, 1.45) },
    { t: 5.6, ...spot(21, 34, 1.3) },
    { t: 9.8, ...spot(22, 34, 1.3) },
  ],
  pointer: [REST],
  clicks: [],
  keys: [{ keys: ["⌘", "S"], t: 9.3 }],
  view: (t) => {
    const { lines, added } = tomlFile(t);
    const cursor = t < 1.9 ? 1 : t < 2.9 ? 6 : t < 3.9 ? 13 : t < 5.4 ? 14 : lines.length - 1;
    return <Editor explorer={EXPLORER} tabs={["sparky.toml"]} open="sparky.toml" lines={lines} added={added} cursor={cursor} t={t} />;
  },
};

/** .env: the Discord bot and the Canvas token. Secrets show as dots. */
const envFile = (t: number): { lines: Code[]; added: number[] } => ({
  lines: [
    code(["SPARKY_DISCORD__TOKEN", "key"], "=", [typed("•".repeat(28), t, 0.4, 0.8), "str"]),
    code(["SPARKY_DISCORD__GUILD_ID", "key"], "=", [t >= 0.95 ? "1290457783312334848" : "0", "num"]),
    ...(t >= 1.2 ? [code(["SPARKY_CANVAS__ACCESS_TOKEN", "key"], "=", [typed("•".repeat(24), t, 1.5, 1.9), "str"])] : []),
    code(["SPARKY_MODEL__BASE_URL", "key"], "=", ["http://localhost:8000/v1", "str"]),
  ],
  added: t >= 1.2 ? [2] : [],
});

export const env: Scene = {
  title: "Add the secrets",
  length: 3.4,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.4, ...spot(1, 22, 1.55) },
    { t: 3.0, ...spot(1, 22, 1.55) },
  ],
  pointer: [REST],
  clicks: [],
  keys: [{ keys: ["⌘", "S"], t: 2.4 }],
  view: (t) => {
    const { lines, added } = envFile(t);
    return (
      <Editor
        explorer={EXPLORER}
        tabs={["sparky.toml", ".env"]}
        open=".env"
        lines={lines}
        added={added}
        cursor={t < 0.9 ? 0 : t < 1.2 ? 1 : 2}
        t={t}
      />
    );
  },
};
