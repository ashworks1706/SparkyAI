import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import DragonHead from "./DragonHead";
import { buildGeometry, lengthAtY, pose, type Box, type Geometry, type Mark } from "./geometry";

/** Where the head sits in the logo, as fractions of its box. */
const LOGO_HEAD = { x: 0.78, y: 0.32 };

/** Length over which the head fades into the logo at the end, in body widths. */
const FADE = 3;

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
    end: { x: e.x + e.w * LOGO_HEAD.x, y: e.y + e.h * LOGO_HEAD.y },
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
 * revealing the body behind it, and fades into the logo at the end.
 */
const DragonTrail = () => {
  const host = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const mask = useRef<SVGPathElement>(null);
  const head = useRef<SVGGElement>(null);
  const [geo, setGeo] = useState<Geometry | null>(null);

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

  const update = useCallback(() => {
    if (!geo || !svg.current || !mask.current || !head.current) return;
    const { track } = geo;
    const view = -svg.current.getBoundingClientRect().top + window.innerHeight * 0.55;
    const target = reducedMotion() ? geo.endY : Math.min(Math.max(view, geo.startY), geo.endY);
    const at = lengthAtY(track, target);

    mask.current.style.strokeDashoffset = `${track.total - at}`;
    const h = pose(track, at);
    const flip = Math.cos((h.angle * Math.PI) / 180) < 0 ? -1 : 1;
    head.current.setAttribute(
      "transform",
      `translate(${h.x.toFixed(1)} ${h.y.toFixed(1)}) rotate(${h.angle.toFixed(1)}) scale(${geo.head} ${geo.head * flip})`,
    );
    head.current.style.opacity = `${Math.min(1, (track.total - at) / (geo.body * FADE))}`;
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
  }, [update]);

  return (
    <div ref={host} aria-hidden className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
      {geo && (
        <svg ref={svg} width={geo.width} height={geo.height} className="absolute left-0 top-0">
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
                pathLength={geo.track.total}
                strokeDasharray={`${geo.track.total} ${geo.track.total}`}
                style={{ strokeDashoffset: geo.track.total }}
              />
            </mask>
          </defs>

          <g mask="url(#dragon-reveal)">
            {geo.legs.map((m, i) => (
              <Leg key={`l${i}`} mark={m} scale={geo.body / 44} />
            ))}
            {geo.fins.map((m, i) => (
              <Fin key={`f${i}`} mark={m} scale={geo.body / 44} />
            ))}
            <path d={geo.d} fill="none" stroke="var(--color-ink)" strokeWidth={geo.body + 5} />
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

          <g ref={head}>
            <DragonHead />
          </g>
        </svg>
      )}
    </div>
  );
};

export default DragonTrail;
