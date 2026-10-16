import type { Scene } from "../../timeline";
import { CAMPUS } from "../campus";
import { EXPLORER, REST, spot, withFile } from "../editor";
import { Editor } from "../kit";
import { charCount, code, firstChars, type Code } from "../text";

/** One method of the LiveSource impl: its signature, a body line, and the closing brace. */
const method = (name: string, ret: string, body: Code): Code[] => [
  code("    ", ["fn", "kw"], " ", [name, "fn"], "(&", ["self", "kw"], ") -> ", [ret, "ty"], " {"),
  [{ text: "        " }, ...body],
  code("    }"),
  [],
];

/** The engine side: the key, the label its button carries, and its parameters. */
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
const STEPS = { rs: 0, catalog: 2.7 };

/** Scene: write the engine side of a live source, then list it in the catalog. Platform publishes the query it runs. */
export const source: Scene = {
  title: "Write a live source of your own",
  length: 4.7,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.3, ...spot(9, 34, 1.2) },
    { t: 1.5, ...spot(17, 34, 1.2) },
    { t: 2.5, ...spot(17, 34, 1.2) },
    { t: 2.9, ...spot(5, 28, 1.5) },
    { t: 4.3, ...spot(5, 28, 1.5) },
  ],
  pointer: [REST],
  clicks: [],
  keys: [
    { keys: ["⌘", "S"], t: 2.3 },
    { keys: ["⌘", "S"], t: 4.2 },
  ],
  view: (t) => {
    const rows = withFile(EXPLORER, "      sources/", "        rec_hours.rs");
    if (t < STEPS.catalog) {
      const lines = firstChars(REC_RS, Math.floor(charCount(REC_RS) * Math.min(1, Math.max(0, (t - 0.2) / 2.0))));
      return <Editor explorer={rows} tabs={["rec_hours.rs"]} open="rec_hours.rs" lines={lines} cursor={lines.length - 1} t={t} />;
    }
    const { lines, added } = catalog(t - STEPS.catalog);
    return <Editor explorer={rows} tabs={["rec_hours.rs", "mod.rs"]} open="mod.rs" lines={lines} added={added} t={t} />;
  },
};
