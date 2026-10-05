import { useEffect, useRef } from "react";
import { useInView, useReducedMotion } from "motion/react";
import Heading from "./Heading";

/** One turn on Discord, recorded from the mockup in src/demo by scripts/record-demo.mjs. */
const VIDEO = { src: "/demo/sparky-demo.mp4", poster: "/demo/sparky-demo.webp", width: 1600, height: 1000 };

/** A recorded turn: a mention opens a thread, the steps tick by, and the answer lands with its sources. */
const InAction = () => {
  const video = useRef<HTMLVideoElement>(null);
  const seen = useInView(video, { margin: "100px 0px" });
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = video.current;
    if (!el || reduced) return;
    if (seen) Promise.resolve().then(() => el.play()).catch(() => undefined);
    else el.pause?.();
  }, [seen, reduced]);

  return (
    <section id="in-action" aria-label="Sparky in action" className="relative z-10 bg-paper py-20">
      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        <Heading title="It shows its working." />
        <div className="mt-12 overflow-hidden rounded-2xl bg-shu shadow-[0_40px_80px_-40px_rgba(20,18,16,0.6)] ring-1 ring-ink/10">
          <video
            ref={video}
            src={VIDEO.src}
            poster={VIDEO.poster}
            width={VIDEO.width}
            height={VIDEO.height}
            muted
            loop
            playsInline
            preload="metadata"
            controls={!!reduced}
            aria-label="Sparky answering a question in a Discord thread"
            className="block h-auto w-full"
          />
        </div>
      </div>
    </section>
  );
};

export default InAction;
