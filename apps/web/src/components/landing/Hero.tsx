import { ArrowRight } from "lucide-react";
import GithubMark from "@/components/brand/GithubMark";
import HeroVisual from "./HeroVisual";
import { REPO } from "./content";

const Hero = () => (
  <section
    aria-label="Introduction"
    className="relative isolate px-5 pb-20 pt-28 sm:px-8 sm:pb-24 sm:pt-32 lg:px-10"
  >
    <div className="mx-auto grid max-w-6xl items-center gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:gap-8">
      <div className="text-center lg:text-left">
        <h1 className="mt-7 text-balance text-[clamp(2.6rem,5.6vw,4.5rem)] font-semibold leading-[0.95] tracking-[-0.045em] text-stone-900">
          Your university
          <span className="block bg-gradient-to-r from-sparky-maroon via-[#b3325c] to-sparky-gold bg-clip-text text-transparent">
            copilot
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-xl text-pretty text-base leading-7 text-stone-600 sm:text-lg sm:leading-8 lg:mx-0">
          Ask about a course, a scholarship, tonight&rsquo;s library hours or
          the next shuttle. Sparky reads the real ASU pages, opens the ones a
          search only pointed at, and answers with the date it read them.
        </p>

        <div className="mt-9 flex flex-col items-center gap-3 sm:flex-row sm:justify-center lg:justify-start">
          <a
            href={`${REPO}#readme`}
            className="group inline-flex h-12 items-center gap-2 rounded-full bg-sparky-maroon px-7 text-sm font-semibold text-white shadow-maroon transition-all hover:-translate-y-0.5 hover:bg-[#7a1937] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sparky-maroon"
          >
            Add Sparky to your server
            <ArrowRight
              className="h-4 w-4 transition-transform group-hover:translate-x-0.5"
              aria-hidden
            />
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-12 items-center gap-2 rounded-full border border-white/70 bg-white/60 px-7 text-sm font-semibold text-stone-800 shadow-glass backdrop-blur-xl transition-all hover:-translate-y-0.5 hover:bg-white/80"
          >
            <GithubMark className="h-4 w-4" />
            Read the source
          </a>
        </div>
      </div>

      <div className="relative h-72 w-full sm:h-96 lg:h-[30rem]">
        <HeroVisual />
      </div>
    </div>
  </section>
);

export default Hero;
