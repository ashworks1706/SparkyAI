import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import App from "./App";

describe("website routes", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/");
  });

  it("leads with what Sparky is and what it costs a student to try", () => {
    render(<App />);

    const main = within(screen.getByRole("main"));
    expect(main.getByRole("heading", { level: 1 })).toHaveTextContent(/your university\s*copilot/i);
    expect(main.getAllByRole("link", { name: /add sparky to your server/i })[0]).toHaveAttribute(
      "href",
      expect.stringContaining("github.com/ashworks1706/SparkyAI"),
    );
  });

  it("shows the product before it asks for anything", () => {
    render(<App />);

    for (const label of [/sparky in action/i, /what students ask/i, /how it works/i, /open source/i]) {
      expect(screen.getByRole("region", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole("region", { name: /project readme/i })).toBeInTheDocument();
  });

  it("names a source under the answer it shows", async () => {
    render(<App />);

    const action = within(screen.getByRole("region", { name: /sparky in action/i }));
    const cite = await action.findByRole(
      "link",
      { name: /asu course catalog/i },
      { timeout: 6000 },
    );
    expect(cite).toHaveAttribute("href", expect.stringContaining("catalog.apps.asu.edu"));
  });

  it("keeps the navigation to the repository and the legacy site", () => {
    render(<App />);

    const nav = within(screen.getByRole("navigation", { name: /primary/i }));
    expect(nav.getByRole("link", { name: /legacy/i })).toHaveAttribute("href", "/old");
    expect(nav.getByRole("link", { name: /github/i })).toHaveAttribute(
      "href",
      "https://github.com/ashworks1706/SparkyAI",
    );
  });

  it("serves the legacy website at old without signup controls", () => {
    window.history.pushState(null, "", "/old");

    render(<App />);

    expect(
      screen.getByRole("heading", { name: /your intelligent university copilot/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /get started|sign up|send message/i }),
    ).not.toBeInTheDocument();
  });
});
