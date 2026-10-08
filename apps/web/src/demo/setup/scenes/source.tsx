import type { Scene } from "../../timeline";
import { CAMPUS } from "../campus";
import { EXPLORER, REST, spot, withFile } from "../editor";
import { Editor } from "../kit";
import { charCount, code, firstChars, type Code } from "../text";

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
