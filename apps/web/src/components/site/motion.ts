import Lenis from "lenis";
import { useEffect } from "react";
import "lenis/dist/lenis.css";

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
