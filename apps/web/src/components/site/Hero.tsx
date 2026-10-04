import { ArrowRight } from "lucide-react";
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
      <div className="lg:pt-24">
        <h1 className="font-serif text-[clamp(3.1rem,8vw,6.5rem)] font-black leading-[0.95] tracking-[-0.035em]">
          {WORDS.map((word, i) => (
            <span key={word} className="block overflow-hidden pb-2">
              <span
                className={`inline-block motion-safe:animate-lift ${i === 2 ? "text-shu" : ""}`}
                style={{ animationDelay: `${i * 0.1}s` }}
              >
                {word}
              </span>{" "}
            </span>
          ))}
        </h1>
        <svg aria-hidden viewBox="0 0 300 14" className="-mt-1 h-4 w-64 sm:w-80">
          <path
            d="M4 9 C 60 2 120 12 180 6 S 270 4 296 8"
            pathLength={1}
            strokeDasharray="1 1"
            fill="none"
            stroke="var(--color-kin)"
            strokeWidth="6"
            strokeLinecap="round"
            className="motion-safe:animate-draw"
            style={{ animationDelay: "0.35s" }}
          />
        </svg>

        <p
          className="mt-7 max-w-lg text-pretty text-lg leading-8 text-ink-soft motion-safe:animate-rise"
          style={{ animationDelay: "0.3s" }}
        >
          Ask about a course, a scholarship, tonight&rsquo;s library hours or the next shuttle.
          Sparky reads the real ASU pages and answers with the source.
        </p>

        <div
          className="mt-10 flex flex-wrap items-center gap-3 motion-safe:animate-rise"
          style={{ animationDelay: "0.4s" }}
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
            GitHub
          </a>
        </div>
      </div>

      <div data-dragon="hero" className="h-[46svh] lg:h-[80svh]" />
    </div>
  </section>
);

export default Hero;
