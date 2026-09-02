import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import App from "./App";

describe("website routes", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/");
  });

  it("renders the new landing statement, logo, and essential navigation", () => {
    render(<App />);

    expect(
      within(screen.getByRole("main")).getByRole("heading", { name: "SparkyAI" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "University Copilot" }),
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole("main")).getByRole("img", {
        name: /sparkyai dragon logo/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: /primary/i })).toBeInTheDocument();
    const nav = within(screen.getByRole("navigation", { name: /primary/i }));
    expect(nav.getByRole("link", { name: /legacy/i })).toHaveAttribute(
      "href",
      "/old",
    );
    expect(nav.getByRole("link", { name: /readme/i })).toHaveAttribute(
      "href",
      "#readme",
    );
    expect(nav.getByRole("link", { name: /github/i })).toHaveAttribute(
      "href",
      "https://github.com/ashworks1706/SparkyAI",
    );
    expect(screen.getByRole("region", { name: /project readme/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /sparkyai v1: a multi-agent university copilot/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /your intelligent university copilot/i }),
    ).not.toBeInTheDocument();
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
