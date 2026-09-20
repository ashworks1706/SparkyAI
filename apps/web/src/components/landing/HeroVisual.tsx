import { Suspense, lazy, useEffect, useState } from "react";

const Scene3D = lazy(() => import("./Scene3D"));

/** Whether this browser will paint WebGL at all. */
const hasWebGL = () => {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
};

const query = (q: string) => window.matchMedia(q);

/**
 * Whether to run the 3D scene: a wide enough viewport, no reduced-motion preference, and WebGL.
 * A phone and a reader who asked for less motion both get the still instead.
 */
const useWantsScene = () => {
  const [wants, setWants] = useState(false);
  useEffect(() => {
    const motion = query("(prefers-reduced-motion: reduce)");
    const width = query("(min-width: 768px)");
    const decide = () =>
      setWants(!motion.matches && width.matches && hasWebGL());
    decide();
    motion.addEventListener("change", decide);
    width.addEventListener("change", decide);
    return () => {
      motion.removeEventListener("change", decide);
      width.removeEventListener("change", decide);
    };
  }, []);
  return wants;
};

/** What stands in for the scene: the same dragon, with no canvas behind it. */
const Still = () => (
  <div className="absolute inset-0 grid place-items-center">
    <img
      src="/brand/sparkyai-logo.png"
      alt="Sparky, the SparkyAI dragon"
      className="h-full w-full object-contain motion-safe:animate-float"
    />
  </div>
);

/** The hero object, with the still shown until the scene is wanted and loaded. */
const HeroVisual = () => {
  const wants = useWantsScene();
  return (
    <div className="absolute inset-0" data-testid="hero-visual">
      {wants ? (
        <Suspense fallback={<Still />}>
          <Scene3D />
        </Suspense>
      ) : (
        <Still />
      )}
    </div>
  );
};

export default HeroVisual;
