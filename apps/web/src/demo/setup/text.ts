/** One line of code: its text split into coloured runs. */
export type Code = { text: string; tone?: Tone }[];

/** A syntax colour. */
export type Tone = "key" | "str" | "num" | "com" | "kw" | "fn" | "ty";

/** Splits source into runs: a line is a list of [text, tone] pairs. */
export const code = (...runs: (string | [string, Tone])[]): Code =>
  runs.map((r) => (typeof r === "string" ? { text: r } : { text: r[0], tone: r[1] }));

/** The first count characters of lines, each line break counting as one. */
export const firstChars = (lines: Code[], count: number): Code[] => {
  const out: Code[] = [];
  let left = count;
  for (const line of lines) {
    if (left < 0) break;
    const kept: Code = [];
    for (const run of line) {
      if (left <= 0) break;
      kept.push({ ...run, text: run.text.slice(0, left) });
      left -= run.text.length;
    }
    out.push(kept);
    left -= 1;
  }
  return out;
};

/** How many characters lines hold, each line break counting as one. */
export const charCount = (lines: Code[]) => lines.reduce((n, line) => n + line.reduce((m, r) => m + r.text.length, 0) + 1, 0);
