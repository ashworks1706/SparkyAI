import { ArrowUpRight } from "lucide-react";
import GithubMark from "./GithubMark";
import Section from "./Section";
import { ARCHITECTURE, REPO } from "./content";

/** What the repository is made of. */
const STACK = [
  { label: "Engine", value: "Rust, the agent loop and its HTTP surface" },
  { label: "Bot", value: "Rust, a Discord client of the engine" },
  { label: "Scraper", value: "Python, ingestion and live source queries" },
  { label: "Stores", value: "PostgreSQL with pgvector, Redis, MinIO" },
  { label: "Model", value: "Any OpenAI-compatible server, llama.cpp by default" },
  { label: "Traces", value: "Phoenix over OTLP, every turn as a tree" },
];

const POINTS = [
  { title: "Open source", body: "Read the loop, the prompt and the policy that gates every write." },
  { title: "Self-hosted", body: "Compose brings up the stack. Your data stays on your machine." },
  { title: "Built by students", body: "For the ACM, AI Society and SoDA communities at ASU." },
];

/** Why and how to run it yourself. */
const OpenSource = () => (
  <Section
    id="open-source"
    label="Open source"
    numeral="四"
    margin="公開"
    title="Run it for your own campus."
    lead="Nothing here is a black box. The sources it reads are declared in one file each, and pointing them at another university is editing those files."
  >
    <div className="grid gap-12 lg:grid-cols-2 lg:items-start">
      <div>
        <dl className="space-y-6">
          {POINTS.map((point) => (
            <div key={point.title} className="border-l-2 border-kin pl-5">
              <dt className="font-serif font-semibold">{point.title}</dt>
              <dd className="mt-1 text-sm leading-6 text-ink-soft">{point.body}</dd>
            </div>
          ))}
        </dl>

        <div className="mt-10 flex flex-wrap gap-3">
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-11 items-center gap-2 rounded-full bg-shu px-6 text-sm font-medium text-paper transition-colors hover:bg-shu-deep"
          >
            <GithubMark className="h-4 w-4" />
            View on GitHub
          </a>
          <a
            href={ARCHITECTURE}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-11 items-center gap-2 rounded-full border border-ink/20 px-6 text-sm font-medium transition-colors hover:border-shu hover:text-shu"
          >
            Architecture
            <ArrowUpRight className="h-4 w-4" aria-hidden />
          </a>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-ink/15 bg-white">
        <dl className="divide-y divide-ink/10">
          {STACK.map((row) => (
            <div key={row.label} className="flex gap-4 px-5 py-3.5 sm:px-6">
              <dt className="w-20 shrink-0 font-mono text-xs uppercase tracking-[0.1em] text-shu">
                {row.label}
              </dt>
              <dd className="text-sm text-ink-soft">{row.value}</dd>
            </div>
          ))}
        </dl>
        <div className="bg-ink px-5 py-4 font-mono text-xs text-paper sm:px-6">
          <span className="select-none text-kin">$ </span>
          just up
          <span className="ml-1 inline-block h-3.5 w-1.5 translate-y-0.5 bg-paper motion-safe:animate-caret" />
        </div>
      </div>
    </div>
  </Section>
);

export default OpenSource;
