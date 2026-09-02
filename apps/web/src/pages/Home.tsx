import { ArrowUpRight, Github } from "lucide-react";
import { Link } from "react-router-dom";
import { BrandMark } from "@/components/brand/BrandLogo";
import ReadmeSection from "@/components/sections/ReadmeSection";

const Home = () => (
  <div className="min-h-screen bg-white text-[#1c1917]">
    <header className="fixed inset-x-0 top-0 z-20 px-4 pt-4 sm:px-8 sm:pt-6">
      <nav
        aria-label="Primary navigation"
        className="mx-auto flex h-12 max-w-6xl items-center justify-between rounded-full border border-stone-200/80 bg-white/70 px-4 shadow-[0_8px_30px_rgba(28,25,23,0.05)] backdrop-blur-xl sm:h-14 sm:px-5"
      >
        <Link to="/" className="flex items-center gap-2" aria-label="SparkyAI home">
          <BrandMark className="h-7 w-auto sm:h-8" decorative />
          <span className="font-editorial text-base font-medium tracking-[-0.03em] sm:text-lg">
            SparkyAI
          </span>
        </Link>

        <div className="flex items-center gap-4 text-xs font-medium text-stone-600 sm:gap-6 sm:text-sm">
          <a href="#readme" className="transition-colors hover:text-[#6f1635]">
            Readme
          </a>
          <Link to="/old" className="transition-colors hover:text-[#6f1635]">
            Legacy
          </Link>
          <a
            href="https://github.com/ashworks1706/SparkyAI/blob/main/docs/ROADMAP.md"
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-1 transition-colors hover:text-[#6f1635] sm:flex"
          >
            Roadmap
            <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
          <a
            href="https://github.com/ashworks1706/SparkyAI"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="transition-colors hover:text-[#6f1635]"
          >
            <Github className="h-4 w-4 sm:h-[18px] sm:w-[18px]" />
          </a>
        </div>
      </nav>
    </header>

    <main className="h-screen px-7 pb-7 pt-20 sm:px-10 sm:pb-10 sm:pt-24 md:px-16 lg:px-24">
      <div className="mx-auto grid h-full max-w-[1400px] grid-rows-[auto_1fr] items-center gap-3 md:grid-cols-[1fr_0.9fr] md:grid-rows-1 md:gap-12 lg:gap-20">
        <div className="z-10 self-end pb-2 md:self-center md:pb-0">
          <p className="mb-3 text-[0.65rem] font-medium uppercase tracking-[0.24em] text-stone-500 sm:text-xs">
            学生生活のための
          </p>
          <h1 className="font-editorial text-[clamp(3rem,6vw,6.5rem)] font-medium leading-[0.88] tracking-[-0.06em]">
            SparkyAI
          </h1>
          <h2 className="font-editorial mt-4 text-[clamp(1.5rem,2.7vw,3.25rem)] font-normal leading-[1.02] tracking-[-0.04em] text-[#6f1635] md:mt-6">
            University Copilot
          </h2>
          <p className="mt-5 max-w-lg text-sm leading-6 text-stone-600 sm:text-base sm:leading-7">
            An open-source agent that watches ASU for you — scholarships,
            events, jobs, and deadlines that match you, with a dated source
            for every answer.
          </p>
        </div>

        <div className="flex min-h-0 items-center justify-center md:h-full md:justify-end">
          <BrandMark className="max-h-[46vh] w-auto max-w-[74vw] object-contain sm:max-h-[48vh] md:max-h-[58vh] md:max-w-[36vw]" />
        </div>
      </div>
    </main>

    <ReadmeSection />
  </div>
);

export default Home;
