import { describe, expect, it } from "vitest";
import { buildGeometry, lengthAtY, pose, smoothPath } from "./geometry";

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
  it("starts off the right edge and comes to rest on the logo", () => {
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

  it("runs straight down the middle below the hero instead of swinging between margins", () => {
    const { track } = buildGeometry(layout);
    const below = layout.hero.y + layout.hero.h + 200;
    for (let i = 0; i < track.ys.length; i++) {
      if (track.ys[i] > below && track.ys[i] < layout.end.y - 200) {
        expect(Math.abs(track.xs[i] - layout.width / 2)).toBeLessThan(40);
      }
    }
  });

  it("draws nothing for fewer than two points", () => {
    expect(smoothPath([{ x: 0, y: 0 }]).d).toBe("");
  });

  it("measures a straight run exactly and finds points along it", () => {
    const { track } = smoothPath([
      { x: 0, y: 0 },
      { x: 0, y: 100 },
    ]);
    expect(track.total).toBeCloseTo(100, 5);
    expect(pose(track, 25)).toMatchObject({ x: 0, y: 25, angle: 90 });
    expect(lengthAtY(track, 60)).toBeGreaterThanOrEqual(60);
    expect(lengthAtY(track, 1000)).toBe(track.total);
  });

  it("places fins along the whole body", () => {
    const geo = buildGeometry(layout);
    expect(geo.fins.length).toBeGreaterThan(geo.track.total / 40);
    expect(geo.legs.length).toBeGreaterThan(0);
  });
});
