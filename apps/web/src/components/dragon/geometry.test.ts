import { describe, expect, it } from "vitest";
import { buildGeometry, smoothPath } from "./geometry";

const layout = {
  width: 1440,
  height: 6000,
  hero: { x: 760, y: 80, w: 560, h: 720 },
  crossings: [1200, 2600, 4000],
  end: { x: 720, y: 5200 },
};

const endpoints = (d: string) => {
  const nums = d.match(/-?\d+(\.\d+)?/g)?.map(Number) ?? [];
  return { start: { x: nums[0], y: nums[1] }, end: { x: nums[nums.length - 2], y: nums[nums.length - 1] } };
};

describe("dragon geometry", () => {
  it("starts off the right edge and comes to rest on the end anchor", () => {
    const { d } = buildGeometry(layout);
    const { start, end } = endpoints(d);
    expect(start.x).toBeGreaterThan(layout.width);
    expect(end).toEqual(layout.end);
  });

  it("keeps the head between the hero and the resting place", () => {
    const geo = buildGeometry(layout);
    expect(geo.startY).toBeGreaterThan(layout.hero.y);
    expect(geo.startY).toBeLessThan(layout.hero.y + layout.hero.h);
    expect(geo.endY).toBe(layout.end.y);
  });

  it("crosses the middle of the page at every divider", () => {
    const { d } = buildGeometry(layout);
    for (const y of layout.crossings) {
      expect(d).toContain(`720.0 ${y.toFixed(1)}`);
    }
  });

  it("draws nothing for fewer than two points", () => {
    expect(smoothPath([{ x: 0, y: 0 }])).toBe("");
  });
});
