import { useEffect, useRef } from "react";
import { useInView, useReducedMotion } from "motion/react";

/** A recorded demo from public/demo: plays muted while on screen, shows controls under reduced motion. */
const DemoVideo = ({ name, label }: { name: string; label: string }) => {
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
    <div className="overflow-hidden rounded-2xl bg-shu shadow-[0_40px_80px_-40px_rgba(20,18,16,0.6)] ring-1 ring-ink/10">
      <video
        ref={video}
        src={`/demo/${name}.mp4`}
        poster={`/demo/${name}.webp`}
        width={1600}
        height={1000}
        muted
        loop
        playsInline
        preload="metadata"
        controls={!!reduced}
        aria-label={label}
        className="block h-auto w-full"
      />
    </div>
  );
};

export default DemoVideo;
