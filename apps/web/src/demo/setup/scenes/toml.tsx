import { typed, type Scene } from "../../timeline";
import { CAMPUS, MCP } from "../campus";
import { EXPLORER, REST, spot } from "../editor";
import { Editor } from "../kit";
import { charCount, code, firstChars, type Code } from "../text";

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
