import {
  ArrowUpRight,
  Database,
  FileSearch,
  Quote,
  TerminalSquare,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { GlassPanel, Section, SectionLabel } from "./Glass";
import { ARCHITECTURE } from "./content";

type Stage = { icon: LucideIcon; step: string; title: string; body: string };

/** The four things a turn does, in the order it does them. */
const STAGES: Stage[] = [
  {
    icon: Database,
    step: "01",
    title: "Read what is stored",
    body: "A scraper keeps dated copies of ASU pages. Retrieval pulls the passages that match, by meaning and by keyword, and hands them to the model before it is asked anything.",
  },
  {
    icon: FileSearch,
    step: "02",
    title: "Fetch what has to be current",
    body: "Hours today, open seats, the next shuttle: a stored copy would be wrong, so those are fetched as the question is asked, and cached only long enough that a hundred students cost one fetch.",
  },
  {
    icon: TerminalSquare,
    step: "03",
    title: "Open the page itself",
    body: "When a search comes back thin or a tool fails, the turn goes to an isolated container that fetches the page and reads it, rather than telling a student to go and look.",
  },
  {
    icon: Quote,
    step: "04",
    title: "Answer with the source",
    body: "Citations are built from the pages it read, never parsed out of what the model wrote, and carry when the page was fetched.",
  },
];

const HowItWorks = () => (
  <Section id="how-it-works" label="How it works">
    <div className="max-w-2xl">
      <SectionLabel>How it works</SectionLabel>
      <h2 className="mt-4 text-balance text-4xl font-semibold tracking-[-0.035em] text-stone-900 sm:text-5xl">
        Grounded by construction, not by asking nicely
      </h2>
      <p className="mt-5 text-pretty leading-7 text-stone-600">
        The loop is written here rather than handed to a framework, so what the
        answer rests on is something you can read.
      </p>
    </div>

    <ol className="mt-12 grid gap-4 md:grid-cols-2">
      {STAGES.map((stage) => {
        const Icon = stage.icon;
        return (
          <li key={stage.step}>
            <GlassPanel interactive className="h-full p-7">
              <div className="flex items-start justify-between gap-4">
                <span className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-sparky-maroon to-[#b3325c] text-white shadow-maroon">
                  <Icon className="h-5 w-5" aria-hidden />
                </span>
                <span className="font-mono text-2xl font-semibold text-stone-900/10">
                  {stage.step}
                </span>
              </div>
              <h3 className="mt-5 text-lg font-semibold tracking-[-0.015em] text-stone-900">
                {stage.title}
              </h3>
              <p className="mt-3 text-sm leading-6 text-stone-600">
                {stage.body}
              </p>
            </GlassPanel>
          </li>
        );
      })}
    </ol>

    <a
      href={ARCHITECTURE}
      target="_blank"
      rel="noreferrer"
      className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-sparky-maroon underline-offset-4 hover:underline"
    >
      Read the architecture
      <ArrowUpRight className="h-4 w-4" aria-hidden />
    </a>
  </Section>
);

export default HowItWorks;
