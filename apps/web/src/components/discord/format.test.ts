import { describe, expect, it } from "vitest";
import { finishedCard, liveResult, runningCard, sourceLabel, threadName, toolDone, toolStarted } from "./format";

describe("reply format", () => {
  it("writes a running tool call as the engine does, arguments sorted by key", () => {
    expect(toolStarted("search_live", { source: "library_hours", query: "Hayden hours today" })).toBe(
      "\u{1f527} `search_live` (query: Hayden hours today, source: library_hours) — searching live",
    );
  });

  it("clips a returned call to 160 characters on one line", () => {
    const line = toolDone("search_live", { query: "q" }, liveResult("ASU Library Hours", "https://lib.asu.edu/hours", "under an hour ago", "x ".repeat(200)));
    const shown = line.split(" → ")[1];
    expect(Array.from(shown).length).toBeLessThanOrEqual(160);
    expect(shown.endsWith("…")).toBe(true);
    expect(shown.startsWith("Live result from ASU Library Hours (https://lib.asu.edu/hours, fetched under an hour ago): x")).toBe(true);
  });

  it("puts the spinner header over the steps while running, and drops it when finished", () => {
    expect(runningCard(["\u{1f914} thinking"], 1)).toBe("◓ **Sparky is working on it…**\n-# \u{1f914} thinking");
    expect(runningCard([], 0, "Draft")).toBe("◐ **Sparky is answering…**\n\nDraft");
    expect(finishedCard(["a", "b"], "Answer.")).toBe("-# a\n-# b\n\nAnswer.");
  });

  it("names threads and buttons within Discord limits", () => {
    expect(threadName("  what   is  open ")).toBe("what is open");
    expect(threadName("")).toBe("Question");
    expect(sourceLabel("a".repeat(50))).toHaveLength(40);
  });
});
