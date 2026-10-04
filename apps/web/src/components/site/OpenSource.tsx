import { motion } from "motion/react";
import { ArrowUpRight } from "lucide-react";
import Gate from "@/components/dragon/Gate";
import GithubMark from "./GithubMark";
import Heading from "./Heading";
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

const Pillar = () => (
  <div aria-hidden className="relative bg-shu">
    <div className="absolute inset-x-[-18%] bottom-0 h-6 bg-ink" />
  </div>
);

/** The way in: a gate with the logo the dragon comes to rest at, with how to run it yourself. */
const OpenSource = () => (
  <section id="open-source" aria-label="Open source" className="relative z-10 pb-24">
    <div className="mx-auto max-w-6xl px-2 sm:px-8">
      <Gate className="w-full" />
      <div className="grid grid-cols-[2%_4.25%_1fr_4.25%_2%] sm:grid-cols-[16%_4.25%_1fr_4.25%_16%]">
        <div />
        <Pillar />
        <div className="relative px-4 pb-16 sm:px-8">
          <div className="flex justify-center pt-10 sm:pt-14">
            <img
              data-dragon="end"
              src="/brand/sparkyai-logo.png"
              alt=""
              width={717}
              height={779}
              className="h-32 w-auto sm:h-44"
            />
          </div>
          <Heading
            title="Run it for your own campus."
            align="center"
          />

          <motion.div
            className="mt-10 flex flex-wrap justify-center gap-3"
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-12 items-center gap-2 rounded-full bg-shu px-7 text-sm font-semibold text-paper shadow-[0_0_0_2px_var(--color-paper),0_0_0_4px_var(--color-kin)] transition-transform hover:-translate-y-0.5"
            >
              <GithubMark className="h-4 w-4" />
              View on GitHub
            </a>
            <a
              href={ARCHITECTURE}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-12 items-center gap-2 rounded-full border-2 border-ink px-7 text-sm font-semibold transition-colors hover:bg-ink hover:text-paper"
            >
              Architecture
              <ArrowUpRight className="h-4 w-4" aria-hidden />
            </a>
          </motion.div>


          <div className="mx-auto mt-14 max-w-2xl overflow-hidden rounded-md border-2 border-ink bg-white">
            <dl className="divide-y divide-ink/10">
              {STACK.map((row) => (
                <div key={row.label} className="flex gap-4 px-5 py-3">
                  <dt className="w-20 shrink-0 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-shu">
                    {row.label}
                  </dt>
                  <dd className="text-sm text-ink-soft">{row.value}</dd>
                </div>
              ))}
            </dl>
            <div className="bg-ink px-5 py-4 font-mono text-xs text-paper">
              <span className="select-none text-kin">$ </span>
              just up
              <span className="ml-1 inline-block h-3.5 w-1.5 translate-y-0.5 bg-paper motion-safe:animate-caret" />
            </div>
          </div>
        </div>
        <Pillar />
        <div />
      </div>
    </div>
  </section>
);

export default OpenSource;
