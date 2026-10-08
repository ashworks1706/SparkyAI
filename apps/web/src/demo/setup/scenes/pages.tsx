import { typed, type Scene } from "../../timeline";
import { CAMPUS } from "../campus";
import { EXPLORER, REST } from "../editor";
import { Editor } from "../kit";
import { code, type Code } from "../text";

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
