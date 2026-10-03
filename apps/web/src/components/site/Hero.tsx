import { ArrowRight } from "lucide-react";
import GithubMark from "./GithubMark";
import Hanko from "./Hanko";
import { REPO, RUN } from "./content";

const FACTS = [
  { term: "Reads", detail: "ASU pages, each with the date it read them" },
  { term: "Fetches", detail: "hours, seats and shuttles live" },
  { term: "Cites", detail: "the page behind every answer" },
];

/** Opening screen: what Sparky is, the dragon, and the two ways in. */
const Hero = () => (
  <section aria-label="Introduction" className="relative overflow-hidden">
    <div className="mx-auto grid max-w-6xl items-center gap-14 px-5 pb-12 pt-32 sm:px-8 sm:pt-36 lg:grid-cols-[1.1fr_0.9fr] lg:gap-10 lg:pb-16">
      <div className="motion-safe:animate-rise">
        <p className="flex items-center gap-3 text-[0.68rem] sm:text-xs font-medium uppercase tracking-[0.18em] text-ink-soft sm:tracking-[0.25em]">
          <Hanko className="h-7 shrink-0 text-sm" />
          <span lang="ja" className="whitespace-nowrap font-serif normal-case tracking-[0.2em] text-ink">
            学生の相棒
          </span>
          <span className="h-px w-6 shrink-0 bg-ink/20" />
          <span className="whitespace-nowrap">ASU student copilot</span>
        </p>

        <h1 className="mt-8 font-serif text-[clamp(2.75rem,6vw,4.75rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
          Your university
          <br />
          <span className="text-shu">copilot</span>
          <span className="text-kin">.</span>
        </h1>

        <p className="mt-7 max-w-lg text-pretty text-lg leading-8 text-ink-soft">
          Ask about a course, a scholarship, tonight&rsquo;s library hours or the next
          shuttle. Sparky reads the real ASU pages, opens the ones a search only pointed
          at, and answers with the date it read them.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <a
            href={RUN}
            target="_blank"
            rel="noreferrer"
            className="group inline-flex h-12 items-center gap-2 rounded-full bg-ink px-7 text-sm font-medium text-paper transition-colors hover:bg-shu"
          >
            Run Sparky
            <ArrowRight
              className="h-4 w-4 transition-transform group-hover:translate-x-0.5"
              aria-hidden
            />
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-12 items-center gap-2 rounded-full border border-ink/20 px-7 text-sm font-medium transition-colors hover:border-shu hover:text-shu"
          >
            <GithubMark className="h-4 w-4" />
            Read the source
          </a>
        </div>

        <dl className="mt-14 grid max-w-lg grid-cols-3 border-t border-ink/10 pt-6">
          {FACTS.map((fact, i) => (
            <div key={fact.term} className={i > 0 ? "border-l border-ink/10 pl-4" : "pr-4"}>
              <dt className="font-serif text-sm font-semibold text-shu">{fact.term}</dt>
              <dd className="mt-1 text-xs leading-5 text-ink-soft">{fact.detail}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="relative mx-auto w-full max-w-md" data-testid="hero-visual">
        <span
          aria-hidden
          className="absolute -right-3 -top-3 h-1/3 w-1/3 rounded-full bg-shu sm:-right-6 sm:-top-6"
        />
        <div className="relative aspect-square rounded-full border border-kin bg-paper p-4">
          <div className="grid h-full w-full place-items-center rounded-full border border-ink/10 bg-white">
            <img
              src="/brand/sparkyai-logo.png"
              alt="Sparky, the SparkyAI dragon"
              className="h-[72%] w-auto"
            />
          </div>
        </div>
        <span
          aria-hidden
          lang="ja"
          className="tategaki absolute -left-1 top-1/2 -translate-y-1/2 font-serif text-sm tracking-[0.5em] text-ink-soft sm:-left-10"
        >
          スパーキー
        </span>
        <Hanko className="absolute bottom-4 left-4 h-14 text-2xl" />
      </div>
    </div>

    <div
      aria-hidden
      className="seigaiha h-20 [mask-image:linear-gradient(to_bottom,transparent,black)]"
    />
  </section>
);

export default Hero;
