/** A point in page coordinates. */
export type Point = { x: number; y: number };

/** A dorsal fin or leg placed along the body: where it sits, its heading, and which side it is on. */
export type Mark = { x: number; y: number; angle: number; side: 1 | -1 };

/** Everything the trail needs to draw, measured from the page layout. */
export type Geometry = {
  width: number;
  height: number;
  d: string;
  body: number;
  head: number;
  /** Page y the head sits at before any scroll. */
  startY: number;
  /** Page y the head reaches at the bottom of the page. */
  endY: number;
};

/** Converts a Catmull-Rom spline through the points into a cubic Bezier path. */
export const smoothPath = (points: Point[]): string => {
  if (points.length < 2) return "";
  const [first] = points;
  let d = `M ${first.x.toFixed(1)} ${first.y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i - 1] ?? points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] ?? p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
};

/** Rect of an element relative to the host element. */
export type Box = { x: number; y: number; w: number; h: number };

/** Inputs measured from the page: the host size, the hero box, crossings, and the resting place. */
export type Layout = {
  width: number;
  height: number;
  hero: Box;
  crossings: number[];
  end: Point;
};

/** Lays the body out: a coil in the hero, then down the margins, crossing the page at each divider. */
export const buildGeometry = ({ width, height, hero, crossings, end }: Layout): Geometry => {
  const narrow = width < 768;
  const body = narrow ? 22 : 44;
  const head = narrow ? 0.32 : 0.54;
  const gutter = narrow ? 10 : Math.max(46, (width - 1152) / 2 - 64);
  const lanes = { right: width - gutter, left: gutter };
  const wiggle = narrow ? 6 : 18;

  const { x, y, w, h } = hero;
  const points: Point[] = [
    { x: width + 160, y: y - 60 },
    { x: x + w * 0.92, y: y + h * 0.02 },
    { x: x + w * 0.3, y: y + h * 0.18 },
    { x: x + w * 0.12, y: y + h * 0.42 },
    { x: x + w * 0.5, y: y + h * 0.56 },
    { x: x + w * 0.86, y: y + h * 0.7 },
  ];
  const startY = y + h * 0.56;

  let side: "left" | "right" = "right";
  let cursor = y + h * 0.95;
  points.push({ x: lanes.right, y: cursor });

  const runLane = (to: number) => {
    const steps = Math.max(1, Math.round((to - cursor) / 260));
    const span = (to - cursor) / steps;
    for (let i = 1; i < steps; i++) {
      const sway = i % 2 === 0 ? wiggle : -wiggle;
      points.push({ x: lanes[side] + (side === "right" ? sway : -sway), y: cursor + span * i });
    }
    cursor = to;
  };

  for (const c of crossings) {
    const reach = narrow ? 90 : 150;
    runLane(c - reach);
    points.push({ x: lanes[side], y: c - reach });
    const next = side === "right" ? "left" : "right";
    points.push({ x: width / 2, y: c });
    points.push({ x: lanes[next], y: c + reach });
    side = next;
    cursor = c + reach;
  }

  const approach = end.y - (narrow ? 120 : 200);
  runLane(approach);
  points.push({ x: lanes[side], y: approach });
  const toward = side === "right" ? 1 : -1;
  points.push({ x: end.x + toward * width * 0.22, y: end.y - 20 });
  points.push({ x: end.x, y: end.y });

  return {
    width,
    height,
    d: smoothPath(points),
    body,
    head,
    startY,
    endY: end.y,
  };
};

/** Samples along a path element, used for placing marks and mapping height to length. */
export type Samples = { lengths: number[]; ys: number[]; total: number };

/** Measures the path every few pixels. */
export const samplePath = (path: SVGPathElement, step = 6): Samples => {
  const total = path.getTotalLength();
  const lengths: number[] = [];
  const ys: number[] = [];
  for (let s = 0; s <= total; s += step) {
    lengths.push(s);
    ys.push(path.getPointAtLength(s).y);
  }
  return { lengths, ys, total };
};

/** First length along the path at which it reaches the given height. */
export const lengthAtY = ({ lengths, ys, total }: Samples, y: number): number => {
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] >= y) return lengths[i];
  }
  return total;
};

/** Position and heading in degrees at a length along the path. */
export const pose = (path: SVGPathElement, s: number, total: number) => {
  const at = Math.min(Math.max(s, 0), total);
  const a = path.getPointAtLength(Math.max(at - 2, 0));
  const b = path.getPointAtLength(Math.min(at + 2, total));
  const p = path.getPointAtLength(at);
  return { x: p.x, y: p.y, angle: (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI };
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
export const placeMarks = (path: SVGPathElement, total: number, offset: number, width: number) => {
  const fins: Mark[] = [];
  const legs: Mark[] = [];
  const at = (s: number, back: boolean): Mark => {
    const { x, y, angle } = pose(path, s, total);
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
