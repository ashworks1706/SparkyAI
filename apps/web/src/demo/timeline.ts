/** Stage size the videos are recorded at. */
export const STAGE = { w: 1600, h: 1000 };

/** A camera or pointer stop: a point of the stage, how far in, and when. */
export type Shot = { t: number; x: number; y: number; s: number };

/** Cubic ease in and out over 0 to 1. */
export const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);

/** The stop at time t, eased between the two stops around it. */
export const at = (stops: Shot[], t: number): Shot => {
  const next = stops.findIndex((s) => s.t > t);
  if (next === -1) return stops[stops.length - 1];
  if (next === 0) return stops[0];
  const a = stops[next - 1];
  const b = stops[next];
  const u = ease((t - a.t) / (b.t - a.t));
  return { t, x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, s: a.s + (b.s - a.s) * u };
};

/** 0 before from, 1 after from + over, eased between. */
export const progress = (t: number, from: number, over: number) => ease(Math.min(1, Math.max(0, (t - from) / over)));

/** How much of text is typed at time t, typing from start to end. */
export const typed = (text: string, t: number, start: number, end: number) =>
  text.slice(0, Math.floor(text.length * Math.min(1, Math.max(0, (t - start) / (end - start)))));

/** How far a press that started at one of clicks has played at time t, 0 when none is playing. */
export const pressAt = (clicks: number[], t: number) =>
  clicks.map((c) => (t >= c && t < c + 0.5 ? (t - c) / 0.5 : 0)).find((p) => p > 0) ?? 0;
