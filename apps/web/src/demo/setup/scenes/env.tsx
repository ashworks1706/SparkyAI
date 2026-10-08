import { typed, type Scene } from "../../timeline";
import { EXPLORER, REST, spot } from "../editor";
import { Editor } from "../kit";
import { code, type Code } from "../text";

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
