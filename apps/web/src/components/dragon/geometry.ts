/** A point in page coordinates. */
export type Point = { x: number; y: number };

/** A dorsal fin or leg placed along the body: where it sits, its heading, and which side it is on. */
export type Mark = { x: number; y: number; angle: number; side: 1 | -1 };

/** The body flattened into points with their running length, for lookups without the SVG API. */
export type Track = { lengths: Float64Array; xs: Float64Array; ys: Float64Array; total: number };

/** Everything the trail needs to draw, measured from the page layout. */
export type Geometry = {
  width: number;
  height: number;
  d: string;
  track: Track;
  fins: Mark[];
  legs: Mark[];
  body: number;
  head: number;
  /** Page y the head sits at before any scroll. */
  startY: number;
  /** Page y the head reaches at the bottom of the page. */
  endY: number;
};

/** Steps each Bezier segment is flattened into. */
const STEPS = 24;

/** Converts a Catmull-Rom spline through the points into a cubic Bezier path and its flattened track. */
export const smoothPath = (points: Point[]): { d: string; track: Track } => {
  const empty = { d: "", track: { lengths: new Float64Array(), xs: new Float64Array(), ys: new Float64Array(), total: 0 } };
  if (points.length < 2) return empty;
  const segments = points.length - 1;
  const n = segments * STEPS + 1;
  const lengths = new Float64Array(n);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const [first] = points;
  xs[0] = first.x;
  ys[0] = first.y;
  let d = `M ${first.x.toFixed(1)} ${first.y.toFixed(1)}`;
  let k = 1;
  for (let i = 0; i < segments; i++) {
    const p0 = points[i - 1] ?? points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] ?? p2;
    const c1x = +(p1.x + (p2.x - p0.x) / 6).toFixed(1);
    const c1y = +(p1.y + (p2.y - p0.y) / 6).toFixed(1);
    const c2x = +(p2.x - (p3.x - p1.x) / 6).toFixed(1);
    const c2y = +(p2.y - (p3.y - p1.y) / 6).toFixed(1);
    const ax = +p1.x.toFixed(1);
    const ay = +p1.y.toFixed(1);
    const bx = +p2.x.toFixed(1);
    const by = +p2.y.toFixed(1);
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${bx.toFixed(1)} ${by.toFixed(1)}`;
    for (let j = 1; j <= STEPS; j++, k++) {
      const t = j / STEPS;
      const u = 1 - t;
      const x = u * u * u * ax + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * bx;
      const y = u * u * u * ay + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * by;
      xs[k] = x;
      ys[k] = y;
      lengths[k] = lengths[k - 1] + Math.hypot(x - xs[k - 1], y - ys[k - 1]);
    }
  }
  return { d, track: { lengths, xs, ys, total: lengths[n - 1] } };
};

/** Rect of an element relative to the host element. */
export type Box = { x: number; y: number; w: number; h: number };

/** Inputs measured from the page: the host size, the hero box, crossings, and the resting place. */
export type Layout = {
  width: number;
  height: number;
  hero: Box;
  crossings: number[];
  /** Point on the logo where the body ends. */
  end: Point;
};

/** Lays the body out: a coil in the hero, then straight down the middle of the page into the logo. */
export const buildGeometry = ({ width, height, hero, crossings, end }: Layout): Geometry => {
  const narrow = width < 768;
  const body = narrow ? 22 : 44;
  const head = narrow ? 0.32 : 0.54;
  const wiggle = narrow ? 6 : 14;
  const lane = width / 2;

  const { x, y, w, h } = hero;
  const points: Point[] = [
    { x: width + 160, y: y - 60 },
    { x: x + w * 0.92, y: y + h * 0.02 },
    { x: x + w * 0.3, y: y + h * 0.18 },
    { x: x + w * 0.12, y: y + h * 0.42 },
    { x: x + w * 0.5, y: y + h * 0.56 },
    { x: x + w * 0.86, y: y + h * 0.7 },
    { x: x + w * 0.5, y: y + h * 0.9 },
    { x: lane, y: y + h + (narrow ? 60 : 120) },
  ];
  const startY = y + h * 0.56;

  const stops = [...crossings, end.y - (narrow ? 120 : 200)].filter((c) => c > points[points.length - 1].y);
  let cursor = points[points.length - 1].y;
  stops.forEach((to, i) => {
    const steps = Math.max(1, Math.round((to - cursor) / 320));
    const span = (to - cursor) / steps;
    for (let k = 1; k <= steps; k++) {
      const sway = (i + k) % 2 === 0 ? wiggle : -wiggle;
      points.push({ x: lane + (k === steps ? 0 : sway), y: cursor + span * k });
    }
    cursor = to;
  });
  points.push({ x: end.x, y: end.y });

  const { d, track } = smoothPath(points);
  const { fins, legs } = placeMarks(track, body / 2 - 2, width);
  return { width, height, d, track, fins, legs, body, head, startY, endY: end.y };
};

/** First length along the track at which it reaches the given height. */
export const lengthAtY = ({ lengths, ys, total }: Track, y: number): number => {
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] >= y) return lengths[i];
  }
  return total;
};

/** Position and heading in degrees at a length along the track. */
export const pose = ({ lengths, xs, ys, total }: Track, s: number) => {
  const at = Math.min(Math.max(s, 0), total);
  let lo = 0;
  let hi = lengths.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (lengths[mid] <= at) lo = mid;
    else hi = mid;
  }
  const span = lengths[hi] - lengths[lo] || 1;
  const f = (at - lengths[lo]) / span;
  const dx = xs[hi] - xs[lo];
  const dy = ys[hi] - ys[lo];
  return { x: xs[lo] + dx * f, y: ys[lo] + dy * f, angle: (Math.atan2(dy, dx) * 180) / Math.PI };
};

/** Which side of the body is its back at a heading: up when running across, outward when running down. */
const backSide = (x: number, angle: number, width: number): 1 | -1 => {
  const r = (angle * Math.PI) / 180;
  const normal = { x: Math.sin(r), y: -Math.cos(r) };
  if (Math.abs(Math.cos(r)) > 0.35) return normal.y < 0 ? 1 : -1;
  const outward = x < width / 2 ? -1 : 1;
  return Math.sign(normal.x) === outward ? 1 : -1;
};

/** Fins along the back and legs along the belly, spaced by length. */
export const placeMarks = (track: Track, offset: number, width: number) => {
  const { total } = track;
  const fins: Mark[] = [];
  const legs: Mark[] = [];
  const at = (s: number, back: boolean): Mark => {
    const { x, y, angle } = pose(track, s);
    const r = (angle * Math.PI) / 180;
    const side = backSide(x, angle, width) * (back ? 1 : -1);
    const k = offset * side;
    return { x: x + Math.sin(r) * k, y: y - Math.cos(r) * k, angle, side: side as 1 | -1 };
  };
  for (let s = 40; s < total - 40; s += 34) fins.push(at(s, true));
  for (let s = 520; s < total - 300; s += 1300) {
    legs.push(at(s, false));
    legs.push(at(s + 150, false));
  }
  return { fins, legs };
};
