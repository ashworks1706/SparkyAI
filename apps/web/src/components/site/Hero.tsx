import { motion } from "motion/react";
import { ArrowDown, ArrowRight } from "lucide-react";
import Embers from "@/components/dragon/Embers";
import Lantern from "@/components/dragon/Lantern";
import GithubMark from "./GithubMark";
import { REPO, RUN } from "./content";

const WORDS = ["Your", "university", "copilot."];

/** Lanterns strung across the top: left offset in percent, cord length, size. */
const LANTERNS = [
  { left: 3, cord: 26, size: 40 },
  { left: 14, cord: 54, size: 50 },
  { left: 27, cord: 20, size: 34 },
  { left: 39, cord: 44, size: 44 },
];

/** Opening screen: the promise, the two ways in, and the dragon coiled beside them. */
const Hero = () => (
  <section aria-label="Introduction" className="relative isolate min-h-[100svh] overflow-hidden">
    <Embers className="absolute inset-0 -z-10 h-full w-full" />

    <div className="absolute inset-x-0 top-16 -z-10 hidden h-48 lg:block">
      <svg aria-hidden className="absolute inset-x-0 top-0 h-10 w-full" preserveAspectRatio="none" viewBox="0 0 100 10">
        <path d="M0 1 Q 25 9 50 2" fill="none" stroke="var(--color-ink)" strokeOpacity=".5" strokeWidth=".15" />
      </svg>
      {LANTERNS.map((l, i) => (
        <div key={l.left} className="absolute top-1" style={{ left: `${l.left}%` }}>
          <Lantern cord={l.cord} size={l.size} delay={i * 1.3} />
        </div>
      ))}
    </div>

    <div className="mx-auto grid max-w-6xl items-center gap-6 px-5 pt-28 sm:px-8 lg:min-h-[100svh] lg:grid-cols-[1.05fr_0.95fr] lg:pt-16">
      <div className="lg:pt-44">
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 rounded-full border border-kin bg-paper/80 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.25em] text-shu"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-shu" />
          The ASU student copilot
        </motion.p>

        <h1 className="mt-7 font-serif text-[clamp(3.1rem,8vw,6.5rem)] font-black leading-[0.95] tracking-[-0.035em]">
          {WORDS.map((word, i) => (
            <span key={word} className="block overflow-hidden pb-2">
              <motion.span
                className={`inline-block ${i === 2 ? "text-shu" : ""}`}
                initial={{ y: "110%" }}
                animate={{ y: 0 }}
                transition={{ duration: 0.9, delay: 0.15 + i * 0.12, ease: [0.22, 1, 0.36, 1] }}
              >
                {word}
              </motion.span>{" "}
            </span>
          ))}
        </h1>
        <motion.svg
          aria-hidden
          viewBox="0 0 300 14"
          className="-mt-1 h-4 w-64 sm:w-80"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <motion.path
            d="M4 9 C 60 2 120 12 180 6 S 270 4 296 8"
            fill="none"
            stroke="var(--color-kin)"
            strokeWidth="6"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.9, delay: 0.8, ease: "easeOut" }}
          />
        </motion.svg>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6 }}
          className="mt-7 max-w-lg text-pretty text-lg leading-8 text-ink-soft"
        >
          Ask about a course, a scholarship, tonight&rsquo;s library hours or the next shuttle.
          Sparky reads the real ASU pages, opens the ones a search only pointed at, and answers
          with the date it read them.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.75 }}
          className="mt-10 flex flex-wrap items-center gap-3"
        >
          <a
            href={RUN}
            target="_blank"
            rel="noreferrer"
            className="group inline-flex h-13 items-center gap-2 rounded-full bg-shu px-8 text-sm font-semibold text-paper shadow-[0_0_0_2px_var(--color-paper),0_0_0_4px_var(--color-kin),0_14px_30px_-10px_rgba(181,24,43,0.6)] transition-transform hover:-translate-y-0.5"
          >
            Run Sparky
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-13 items-center gap-2 rounded-full border-2 border-ink px-7 text-sm font-semibold transition-colors hover:bg-ink hover:text-paper"
          >
            <GithubMark className="h-4 w-4" />
            Read the source
          </a>
        </motion.div>
      </div>

      <div data-dragon="hero" className="h-[46svh] lg:h-[80svh]" />
    </div>

    <a
      href="#in-action"
      className="absolute bottom-6 left-1/2 hidden -translate-x-1/2 flex-col items-center gap-2 text-[0.65rem] font-semibold uppercase tracking-[0.3em] text-ink-soft lg:flex"
    >
      Scroll to wake the dragon
      <ArrowDown className="h-4 w-4 text-shu motion-safe:animate-bounce" aria-hidden />
    </a>
  </section>
);

export default Hero;
