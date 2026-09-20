import { ArrowUpRight, Github, Scale, ServerCog, Users } from "lucide-react";
import { GlassPanel, Section, SectionLabel } from "./Glass";
import { ARCHITECTURE, REPO } from "./content";

/** What the repository is made of. Named because a reader deciding to self-host wants to know. */
const STACK = [
  { label: "Engine", value: "Rust, the agent loop and its HTTP surface" },
  { label: "Bot", value: "Rust, a Discord client of the engine" },
  { label: "Scraper", value: "Python, ingestion and live source queries" },
  { label: "Stores", value: "PostgreSQL with pgvector, Redis, MinIO" },
  {
    label: "Model",
    value: "Any OpenAI-compatible server, llama.cpp by default",
  },
  { label: "Traces", value: "Phoenix over OTLP, every turn as a tree" },
];

const POINTS = [
  {
    icon: Scale,
    title: "Open source",
    body: "Read the loop, the prompt and the policy that gates every write.",
  },
  {
    icon: ServerCog,
    title: "Self-hosted",
    body: "Compose brings up the stack. Your data stays on your machine.",
  },
  {
    icon: Users,
    title: "Built by students",
    body: "For the ACM, AI Society and SoDA communities at ASU.",
  },
];

const OpenSource = () => (
  <Section id="open-source" label="Open source">
    <div className="grid gap-10 lg:grid-cols-[1fr_1fr] lg:items-start lg:gap-16">
      <div>
        <SectionLabel>Open source</SectionLabel>
        <h2 className="mt-4 text-balance text-4xl font-semibold tracking-[-0.035em] text-stone-900 sm:text-5xl">
          Run it for your own campus
        </h2>
        <p className="mt-5 text-pretty leading-7 text-stone-600">
          Nothing here is a black box. The sources it reads are declared in one
          file each, and pointing them at another university is editing those
          files.
        </p>

        <div className="mt-8 space-y-4">
          {POINTS.map((point) => {
            const Icon = point.icon;
            return (
              <div key={point.title} className="flex gap-4">
                <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-sparky-maroon/10 text-sparky-maroon">
                  <Icon className="h-4 w-4" aria-hidden />
                </span>
                <div>
                  <h3 className="text-sm font-semibold text-stone-900">
                    {point.title}
                  </h3>
                  <p className="mt-1 text-sm leading-6 text-stone-600">
                    {point.body}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-9 flex flex-wrap gap-3">
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-11 items-center gap-2 rounded-full bg-stone-900 px-6 text-sm font-semibold text-white transition-all hover:-translate-y-0.5 hover:bg-stone-800"
          >
            <Github className="h-4 w-4" aria-hidden />
            View on GitHub
          </a>
          <a
            href={ARCHITECTURE}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-11 items-center gap-2 rounded-full border border-white/70 bg-white/60 px-6 text-sm font-semibold text-stone-800 shadow-glass backdrop-blur-xl transition-all hover:-translate-y-0.5 hover:bg-white/80"
          >
            Architecture
            <ArrowUpRight className="h-4 w-4" aria-hidden />
          </a>
        </div>
      </div>

      <GlassPanel className="overflow-hidden">
        <div className="flex items-center gap-2 border-b border-white/60 bg-white/40 px-5 py-3.5">
          <span className="h-2.5 w-2.5 rounded-full bg-sparky-maroon/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-sparky-gold/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-stone-300" />
          <span className="ml-2 font-mono text-[0.7rem] text-stone-400">
            the stack
          </span>
        </div>
        <dl className="divide-y divide-stone-200/60">
          {STACK.map((row) => (
            <div key={row.label} className="flex gap-4 px-5 py-3.5 sm:px-6">
              <dt className="w-24 shrink-0 font-mono text-xs uppercase tracking-[0.1em] text-sparky-maroon">
                {row.label}
              </dt>
              <dd className="text-sm text-stone-600">{row.value}</dd>
            </div>
          ))}
        </dl>
        <div className="border-t border-stone-200/60 bg-stone-900 px-5 py-4 font-mono text-xs text-stone-300 sm:px-6">
          <span className="select-none text-sparky-gold">$ </span>
          just up
          <span className="ml-1 inline-block h-3.5 w-1.5 translate-y-0.5 bg-sparky-gold/80 motion-safe:animate-caret" />
        </div>
      </GlassPanel>
    </div>
  </Section>
);

export default OpenSource;
