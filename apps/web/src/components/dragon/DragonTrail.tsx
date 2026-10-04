import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import DragonHead from "./DragonHead";
import Pearl from "./Pearl";
import {
  buildGeometry,
  lengthAtY,
  placeMarks,
  pose,
  samplePath,
  type Box,
  type Geometry,
  type Mark,
  type Samples,
} from "./geometry";

/** Space kept between the head and the pearl it chases, in body widths. */
const PEARL_LEAD = 3.4;

const reducedMotion = () =>
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Rect of an element relative to the host. */
const boxOf = (el: Element, host: DOMRect): Box => {
  const r = el.getBoundingClientRect();
  return { x: r.left - host.left, y: r.top - host.top, w: r.width, h: r.height };
};

/** Measures the anchors the page marks with data-dragon and lays out the body. */
const measure = (host: HTMLElement): Geometry | null => {
  const rect = host.getBoundingClientRect();
  const hero = host.parentElement?.querySelector("[data-dragon=hero]");
  const end = host.parentElement?.querySelector("[data-dragon=end]");
  if (!hero || !end || rect.width === 0) return null;
  const crossings = Array.from(host.parentElement?.querySelectorAll("[data-dragon=cross]") ?? []).map(
    (el) => {
      const b = boxOf(el, rect);
      return b.y + b.h / 2;
    },
  );
  const e = boxOf(end, rect);
  return buildGeometry({
    width: rect.width,
    height: rect.height,
    hero: boxOf(hero, rect),
    crossings,
    end: { x: e.x + e.w / 2, y: e.y + e.h / 2 },
  });
};

const Fin = ({ mark, scale }: { mark: Mark; scale: number }) => (
  <path
    d="M -10 3 Q -7 -10 7 -17 Q 2 -6 10 3 Z"
    transform={`translate(${mark.x.toFixed(1)} ${mark.y.toFixed(1)}) rotate(${mark.angle.toFixed(1)}) scale(${scale} ${scale * mark.side})`}
    fill="var(--color-kin)"
    stroke="var(--color-shu-deep)"
    strokeWidth="1.5"
    strokeLinejoin="round"
  />
);

const Leg = ({ mark, scale }: { mark: Mark; scale: number }) => (
  <g
    transform={`translate(${mark.x.toFixed(1)} ${mark.y.toFixed(1)}) rotate(${mark.angle.toFixed(1)}) scale(${scale} ${-scale * mark.side})`}
  >
    <path
      d="M -14 -6 C -16 12 -4 22 -20 36 L -10 40 C 6 26 4 12 8 -4 Z"
      fill="var(--color-shu)"
      stroke="var(--color-ink)"
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <path
      d="M -20 36 c -8 2 -10 8 -8 12 M -16 39 c -4 4 -4 10 0 13 M -11 40 c 0 5 2 9 6 11"
      fill="none"
      stroke="var(--color-kin)"
      strokeWidth="3"
      strokeLinecap="round"
    />
  </g>
);

/**
 * The dragon that winds down the page. Its head follows the scroll position along the body,
 * chasing the flaming pearl, and the body behind it is revealed as it goes.
 */
const DragonTrail = () => {
  const host = useRef<HTMLDivElement>(null);
  const path = useRef<SVGPathElement>(null);
  const mask = useRef<SVGPathElement>(null);
  const head = useRef<SVGGElement>(null);
  const pearl = useRef<SVGGElement>(null);
  const samples = useRef<Samples | null>(null);
  const [geo, setGeo] = useState<Geometry | null>(null);
  const [marks, setMarks] = useState<{ fins: Mark[]; legs: Mark[] }>({ fins: [], legs: [] });

  useLayoutEffect(() => {
    const node = host.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    let frame = 0;
    const relayout = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setGeo(measure(node)));
    };
    const observer = new ResizeObserver(relayout);
    observer.observe(node);
    window.addEventListener("resize", relayout);
    relayout();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", relayout);
      cancelAnimationFrame(frame);
    };
  }, []);

  useLayoutEffect(() => {
    const p = path.current;
    if (!geo || !p || typeof p.getTotalLength !== "function") return;
    samples.current = samplePath(p);
    setMarks(placeMarks(p, samples.current.total, geo.body / 2 - 2, geo.width));
  }, [geo]);

  const update = useCallback(() => {
    const p = path.current;
    const s = samples.current;
    if (!geo || !p || !s || !mask.current || !head.current || !pearl.current) return;
    const lead = geo.body * PEARL_LEAD;
    const host = path.current.ownerSVGElement?.getBoundingClientRect();
    const view = host ? -host.top + window.innerHeight * 0.55 : geo.endY;
    const target = reducedMotion() ? geo.endY : Math.min(Math.max(view, geo.startY), geo.endY);
    const at = Math.min(lengthAtY(s, target), s.total - lead);

    mask.current.style.strokeDasharray = `${s.total} ${s.total}`;
    mask.current.style.strokeDashoffset = `${s.total - at}`;
    const h = pose(p, at, s.total);
    const flip = Math.cos((h.angle * Math.PI) / 180) < 0 ? -1 : 1;
    head.current.setAttribute(
      "transform",
      `translate(${h.x.toFixed(1)} ${h.y.toFixed(1)}) rotate(${h.angle.toFixed(1)}) scale(${geo.head} ${geo.head * flip})`,
    );
    const ahead = pose(p, at + lead, s.total);
    pearl.current.setAttribute("transform", `translate(${ahead.x.toFixed(1)} ${ahead.y.toFixed(1)})`);
  }, [geo]);

  useEffect(() => {
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(update);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(frame);
    };
  }, [update, marks]);

  return (
    <div ref={host} aria-hidden className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
      {geo && (
        <svg width={geo.width} height={geo.height} className="absolute left-0 top-0">
          <defs>
            <pattern id="dragon-scales" width="18" height="12" patternUnits="userSpaceOnUse">
              <path
                d="M0 12 a9 9 0 0 1 18 0 M-9 6 a9 9 0 0 1 18 0 M9 6 a9 9 0 0 1 18 0"
                fill="none"
                stroke="var(--color-kin)"
                strokeOpacity=".55"
                strokeWidth="1.4"
              />
            </pattern>
            <mask id="dragon-reveal" maskUnits="userSpaceOnUse" x="0" y="0" width={geo.width} height={geo.height}>
              <path
                ref={mask}
                d={geo.d}
                fill="none"
                stroke="white"
                strokeWidth={geo.body * 3}
                style={{ strokeDasharray: 1e6, strokeDashoffset: 1e6 }}
              />
            </mask>
          </defs>

          <g mask="url(#dragon-reveal)">
            {marks.legs.map((m, i) => (
              <Leg key={`l${i}`} mark={m} scale={geo.body / 44} />
            ))}
            {marks.fins.map((m, i) => (
              <Fin key={`f${i}`} mark={m} scale={geo.body / 44} />
            ))}
            <path ref={path} d={geo.d} fill="none" stroke="var(--color-ink)" strokeWidth={geo.body + 5} />
            <path d={geo.d} fill="none" stroke="var(--color-shu)" strokeWidth={geo.body} />
            <path d={geo.d} fill="none" stroke="url(#dragon-scales)" strokeWidth={geo.body - 6} />
            <path
              d={geo.d}
              fill="none"
              stroke="var(--color-kin)"
              strokeWidth={geo.body * 0.26}
              strokeDasharray={`${geo.body * 0.3} ${geo.body * 0.08}`}
            />
          </g>

          <g ref={pearl}>
            <Pearl />
          </g>
          <g ref={head}>
            <DragonHead />
          </g>
        </svg>
      )}
    </div>
  );
};

export default DragonTrail;
