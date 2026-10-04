import { motion } from "motion/react";
import { ArrowUpRight } from "lucide-react";
import Heading from "./Heading";
import { ARCHITECTURE } from "./content";

type Stage = { numeral: string; title: string };

/** The four things a turn does, in order. */
const STAGES: Stage[] = [
  { numeral: "一", title: "Read what is stored" },
  { numeral: "二", title: "Fetch what must be current" },
  { numeral: "三", title: "Open the page itself" },
  { numeral: "四", title: "Answer with the source" },
];

/** A round seal: red disc, gold rings, numeral in the middle. */
const Medallion = ({ numeral }: { numeral: string }) => (
  <svg viewBox="0 0 120 120" className="h-28 w-28" aria-hidden>
    <circle cx="60" cy="60" r="58" fill="var(--color-kin)" />
    <circle cx="60" cy="60" r="52" fill="var(--color-shu)" />
    <circle cx="60" cy="60" r="44" fill="none" stroke="var(--color-kin)" strokeWidth="1.5" strokeDasharray="3 4" />
    <text
      x="60"
      y="62"
      textAnchor="middle"
      dominantBaseline="middle"
      className="font-serif"
      fontSize="44"
      fontWeight="700"
      fill="var(--color-paper)"
    >
      {numeral}
    </text>
  </svg>
);

/** The turn, stage by stage, strung on a cord that draws itself in. */
const HowItWorks = () => (
  <section id="how-it-works" aria-label="How it works" className="relative z-10 py-10">
    <div className="mx-auto max-w-6xl px-5 sm:px-8">
      <Heading
        title="Grounded by construction, not by asking nicely."
        align="center"
      />

      <div className="relative mt-20">
        <svg aria-hidden viewBox="0 0 1000 60" preserveAspectRatio="none" className="absolute inset-x-[12%] top-10 hidden h-16 w-[76%] lg:block">
          <motion.path
            d="M0 30 C 120 -10 210 70 333 30 S 550 -10 666 30 S 880 70 1000 30"
            fill="none"
            stroke="var(--color-ink)"
            strokeWidth="2"
            strokeDasharray="6 8"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            viewport={{ once: true, margin: "-20% 0px" }}
            transition={{ duration: 1.6, ease: "easeInOut" }}
          />
        </svg>
        <ol className="relative grid gap-14 sm:grid-cols-2 lg:grid-cols-4 lg:gap-8">
          {STAGES.map((stage, i) => (
            <motion.li
              key={stage.title}
              className="flex flex-col items-center text-center"
              initial={{ opacity: 0, y: 40, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: "-15% 0px" }}
              transition={{ duration: 0.7, delay: i * 0.18, ease: [0.22, 1, 0.36, 1] }}
            >
              <motion.div whileHover={{ rotate: 12, scale: 1.06 }} transition={{ type: "spring", stiffness: 260, damping: 14 }}>
                <Medallion numeral={stage.numeral} />
              </motion.div>
              <h3 className="mt-6 font-serif text-xl font-bold">{stage.title}</h3>
            </motion.li>
          ))}
        </ol>
      </div>

      <div className="mt-14 text-center">
        <a
          href={ARCHITECTURE}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 border-b-2 border-shu pb-0.5 text-sm font-semibold text-shu transition-colors hover:border-ink hover:text-ink"
        >
          Read the architecture
          <ArrowUpRight className="h-4 w-4" aria-hidden />
        </a>
      </div>
    </div>
  </section>
);

export default HowItWorks;
