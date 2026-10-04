import { useLayoutEffect, useRef, useState } from "react";
import { motion, useScroll, useTransform } from "motion/react";
import Heading from "./Heading";
import { usePinned } from "./motion";

type Source = { title: string; ask: string; source: string };

/** What students ask, and the source that answers each one. */
const SOURCES: Source[] = [
  {
    title: "Courses",
    ask: "what do I need before CSE 310?",
    source: "Class Search",
  },
  {
    title: "Scholarships",
    ask: "scholarships for a CS junior",
    source: "Scholarship Search",
  },
  {
    title: "Library hours",
    ask: "when does Hayden close tonight?",
    source: "ASU Library",
  },
  {
    title: "Shuttles",
    ask: "next shuttle to Poly",
    source: "Shuttle Tracker",
  },
  {
    title: "Study rooms",
    ask: "a room for four at 6pm",
    source: "Library rooms",
  },
  {
    title: "Dining",
    ask: "what's open on Tempe right now?",
    source: "Sun Devil Dining",
  },
  {
    title: "Clubs",
    ask: "is there an AI club?",
    source: "Sun Devil Central",
  },
  {
    title: "Events",
    ask: "anything on this weekend?",
    source: "Events, Registrar",
  },
];

/** A banner hanging from the rope: red head, paper body, gold tassel. */
const Banner = ({ item, i }: { item: Source; i: number }) => (
  <li
    className="group relative w-64 shrink-0 snap-start origin-top motion-safe:animate-[swing_6s_ease-in-out_infinite] sm:w-72"
    style={{ animationDelay: `${-i * 0.7}s` }}
  >
    <div className="mx-auto flex w-40 justify-between">
      <span className="h-8 w-px bg-ink/50" />
      <span className="h-8 w-px bg-ink/50" />
    </div>
    <div className="overflow-hidden rounded-b-md border-2 border-ink bg-white shadow-[0_24px_40px_-24px_rgba(20,18,16,0.4)] transition-transform duration-300 group-hover:-translate-y-1">
      <div className="flex items-center justify-between bg-shu px-5 py-4 text-paper">
        <h3 className="font-serif text-xl font-bold">{item.title}</h3>
        <span className="font-serif text-sm text-kin">{String(i + 1).padStart(2, "0")}</span>
      </div>
      <div className="h-1.5 bg-kin" />
      <div className="flex min-h-36 flex-col p-5">
        <p className="font-mono text-xs leading-5 text-shu">&ldquo;{item.ask}&rdquo;</p>
        <p className="mt-auto border-t border-dashed border-ink/20 pt-3 text-[0.68rem] font-semibold uppercase tracking-[0.18em]">
          {item.source}
        </p>
      </div>
    </div>
    <div className="mx-auto h-6 w-px bg-kin" />
    <div className="mx-auto h-5 w-3 rounded-b-full bg-kin" />
  </li>
);

/** The sources Sparky reaches, as banners that slide past while the section is pinned. */
const Sources = () => {
  const ref = useRef<HTMLElement>(null);
  const track = useRef<HTMLUListElement>(null);
  const pinned = usePinned();
  const [distance, setDistance] = useState(0);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const x = useTransform(scrollYProgress, [0.05, 0.95], [0, -distance]);

  useLayoutEffect(() => {
    const el = track.current;
    if (!el || !pinned) return;
    const measure = () => setDistance(Math.max(0, el.scrollWidth - window.innerWidth + 64));
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [pinned]);

  const list = (
    <>
      <div aria-hidden className="absolute inset-x-0 top-0 h-1 rounded-full bg-ink/70" />
      {SOURCES.map((item, i) => (
        <Banner key={item.title} item={item} i={i} />
      ))}
    </>
  );

  return (
    <section
      id="sources"
      ref={ref}
      aria-label="What students ask"
      className={`relative z-10 ${pinned ? "h-[300vh]" : "py-10"}`}
    >
      <div className={pinned ? "sticky top-0 flex h-screen flex-col justify-center overflow-hidden" : ""}>
        <div className="mx-auto w-full max-w-6xl px-5 sm:px-8">
          <Heading
            title="Everything you would otherwise go digging for."
          />
        </div>
        {pinned ? (
          <motion.ul ref={track} style={{ x }} className="relative mt-12 flex w-max gap-8 pl-[max(1.25rem,calc((100vw-72rem)/2+2rem))] pr-16">
            {list}
          </motion.ul>
        ) : (
          <ul ref={track} className="relative mt-10 flex snap-x gap-6 overflow-x-auto px-5 pb-6">
            {list}
          </ul>
        )}
      </div>
    </section>
  );
};

export default Sources;
