import Lenis from "lenis";
import { useEffect, useState } from "react";
import "lenis/dist/lenis.css";

/** Whether the viewport is at least this wide and the reader has not asked for less motion. */
export const usePinned = (minWidth = 768) => {
  const query = `(min-width: ${minWidth}px) and (prefers-reduced-motion: no-preference)`;
  const [pinned, setPinned] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const list = window.matchMedia(query);
    const change = () => setPinned(list.matches);
    change();
    list.addEventListener("change", change);
    return () => list.removeEventListener("change", change);
  }, [query]);
  return pinned;
};

/** Eases wheel and anchor scrolling on the window while motion is allowed. */
export const useSmoothScroll = () => {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const lenis = new Lenis({
      autoRaf: true,
      lerp: 0.09,
      anchors: { offset: -64 },
    });
    return () => lenis.destroy();
  }, []);
};
