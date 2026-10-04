import { useEffect, useRef } from "react";

type Ember = { x: number; y: number; r: number; vy: number; vx: number; life: number; gold: boolean };

/** Sparks drifting up from the bottom of their box, like a festival fire. */
const Embers = ({ count = 46, className = "" }: { count?: number; className?: string }) => {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext?.("2d");
    if (!el || !ctx || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let frame = 0;
    let w = 0;
    let h = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const resize = () => {
      w = el.clientWidth;
      h = el.clientHeight;
      el.width = w * dpr;
      el.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const spawn = (anywhere: boolean): Ember => ({
      x: Math.random() * w,
      y: anywhere ? Math.random() * h : h + 10,
      r: 0.8 + Math.random() * 2.2,
      vy: 0.3 + Math.random() * 0.9,
      vx: (Math.random() - 0.5) * 0.4,
      life: 0,
      gold: Math.random() > 0.35,
    });
    resize();
    const sparks = Array.from({ length: count }, () => spawn(true));
    const tick = () => {
      ctx.clearRect(0, 0, w, h);
      for (let i = 0; i < sparks.length; i++) {
        const s = sparks[i];
        s.life += 1;
        s.y -= s.vy;
        s.x += s.vx + Math.sin((s.life + i * 20) / 30) * 0.3;
        const fade = Math.min(1, s.y / (h * 0.6));
        if (s.y < -10 || fade <= 0) sparks[i] = spawn(false);
        ctx.beginPath();
        ctx.fillStyle = s.gold ? `rgba(242,183,5,${0.75 * fade})` : `rgba(120,28,53,${0.6 * fade})`;
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      frame = requestAnimationFrame(tick);
    };
    tick();
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
    };
  }, [count]);

  return <canvas ref={canvas} aria-hidden className={`pointer-events-none ${className}`} />;
};

export default Embers;
