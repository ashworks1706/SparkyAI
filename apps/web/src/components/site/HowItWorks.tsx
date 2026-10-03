import { ArrowUpRight } from "lucide-react";
import Section from "./Section";
import { ARCHITECTURE } from "./content";

type Stage = { numeral: string; title: string; body: string };

/** The four things a turn does, in order. */
const STAGES: Stage[] = [
  {
    numeral: "壱",
    title: "Read what is stored",
    body: "A scraper keeps dated copies of ASU pages. Retrieval pulls the passages that match, by meaning and by keyword, before the model is asked anything.",
  },
  {
    numeral: "弐",
    title: "Fetch what must be current",
    body: "Hours today, open seats, the next shuttle are fetched as the question is asked, and cached only long enough that a hundred students cost one fetch.",
  },
  {
    numeral: "参",
    title: "Open the page itself",
    body: "When a search comes back thin or a tool fails, an isolated container fetches the page and reads it, rather than telling a student to go and look.",
  },
  {
    numeral: "肆",
    title: "Answer with the source",
    body: "Citations are built from the pages it read, never parsed out of what the model wrote, and carry when the page was fetched.",
  },
];

/** The turn, stage by stage. */
const HowItWorks = () => (
  <Section
    id="how-it-works"
    label="How it works"
    numeral="三"
    margin="仕組み"
    title="Grounded by construction, not by asking nicely."
    lead="The loop is written here rather than handed to a framework, so what the answer rests on is something you can read."
  >
    <ol className="grid gap-10 md:grid-cols-2 lg:grid-cols-4 lg:gap-8">
      {STAGES.map((stage) => (
        <li key={stage.title} className="border-t-2 border-ink pt-6">
          <span lang="ja" className="font-serif text-4xl font-semibold text-shu">
            {stage.numeral}
          </span>
          <h3 className="mt-5 font-serif text-lg font-semibold">{stage.title}</h3>
          <p className="mt-3 text-sm leading-6 text-ink-soft">{stage.body}</p>
        </li>
      ))}
    </ol>

    <a
      href={ARCHITECTURE}
      target="_blank"
      rel="noreferrer"
      className="mt-12 inline-flex items-center gap-1.5 border-b border-shu pb-0.5 text-sm font-medium text-shu transition-colors hover:border-ink hover:text-ink"
    >
      Read the architecture
      <ArrowUpRight className="h-4 w-4" aria-hidden />
    </a>
  </Section>
);

export default HowItWorks;
