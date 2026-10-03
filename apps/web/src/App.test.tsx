import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import App from "./App";

const REPO = "https://github.com/ashworks1706/SparkyAI";

describe("website routes", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/");
  });

  it("leads with what Sparky is and how to run it", () => {
    render(<App />);

    const main = within(screen.getByRole("main"));
    expect(main.getByRole("heading", { level: 1 })).toHaveTextContent(/your university\s*copilot/i);
    expect(main.getByRole("link", { name: /run sparky/i })).toHaveAttribute(
      "href",
      `${REPO}#run-it`,
    );
  });

  it("shows every section of the product", () => {
    render(<App />);

    for (const label of [/sparky in action/i, /what students ask/i, /how it works/i, /open source/i]) {
      expect(screen.getByRole("region", { name: label })).toBeInTheDocument();
    }
  });

  it("names a source under the answer it shows", async () => {
    render(<App />);

    const action = within(screen.getByRole("region", { name: /sparky in action/i }));
    const cite = await action.findByRole("link", { name: /asu course catalog/i }, { timeout: 6000 });
    expect(cite).toHaveAttribute("href", expect.stringContaining("catalog.apps.asu.edu"));
  });

  it("links the navigation to each section and the repository", () => {
    render(<App />);

    const nav = within(screen.getByRole("navigation", { name: /primary/i }));
    for (const id of ["in-action", "sources", "how-it-works", "open-source"]) {
      expect(document.getElementById(id)).not.toBeNull();
      expect(nav.getByRole("link", { name: new RegExp(id.replace(/-/g, " "), "i") })).toHaveAttribute(
        "href",
        `#${id}`,
      );
    }
    expect(nav.getByRole("link", { name: /github/i })).toHaveAttribute("href", REPO);
  });

  it("serves a not found page for unknown paths, including the retired legacy site", () => {
    window.history.pushState(null, "", "/old");

    render(<App />);

    expect(screen.getByRole("heading", { name: /page not found/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /back to home/i })).toHaveAttribute("href", "/");
  });
});
