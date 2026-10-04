import { useRef, type ReactNode } from "react";
import { motion, useMotionValue, useScroll, useTransform, type MotionValue } from "motion/react";
import { ArrowUpRight, BookOpen, Search, Terminal } from "lucide-react";
import Heading from "./Heading";
import { usePinned } from "./motion";

type Step = { label: string };

/** What happens in one turn, in order, as the reader scrolls. */
const STEPS: Step[] = [
  { label: "Asked" },
  { label: "Read stored pages" },
  { label: "Searched live" },
  { label: "Opened the page" },
  { label: "Answered with the source" },
];

const CITE = { label: "ASU Course Catalog", href: "https://catalog.apps.asu.edu/catalog/courses" };

/** Range of scroll progress over which step i appears. */
const span = (i: number): [number, number] => {
  const at = 0.08 + i * 0.17;
  return [at, at + 0.08];
};

/** A block that fades and rises in over its step's range of scroll progress. */
const Appear = ({ progress, i, children }: { progress: MotionValue<number>; i: number; children: ReactNode }) => {
  const range = span(i);
  const opacity = useTransform(progress, range, [0, 1]);
  const y = useTransform(progress, range, [14, 0]);
  return <motion.div style={{ opacity, y }}>{children}</motion.div>;
};

/** One row of the step list, lit while its step is the latest shown. */
const StepRow = ({ progress, i, step }: { progress: MotionValue<number>; i: number; step: Step }) => {
  const [start] = span(i);
  const next = i + 1 < STEPS.length ? span(i + 1)[0] : 1.01;
  const color = useTransform(progress, [start - 0.01, start, next - 0.01, next], [
    "rgba(20,18,16,0.3)",
    "rgba(20,18,16,1)",
    "rgba(20,18,16,1)",
    "rgba(20,18,16,0.45)",
  ]);
  const dot = useTransform(progress, [start - 0.01, start], [0.25, 1]);
  return (
    <li className="flex gap-4">
      <motion.span style={{ opacity: dot }} className="mt-1.5 h-3 w-3 shrink-0 rotate-45 bg-shu" />
      <motion.div style={{ color }}>
        <p className="font-serif text-lg font-bold">{step.label}</p>
      </motion.div>
    </li>
  );
};

/** The hanging scroll that holds the turn. */
const ScrollPaper = ({ progress }: { progress: MotionValue<number> }) => (
  <div className="relative mx-auto w-full max-w-xl">
    <div className="relative z-10 mx-auto flex h-5 w-[108%] -translate-x-[3.7%] items-center justify-between rounded-full bg-ink px-1">
      <span className="h-6 w-6 rounded-full bg-kin" />
      <span className="h-6 w-6 rounded-full bg-kin" />
    </div>
    <div className="relative -mt-1 border-x-[10px] border-shu bg-white px-6 py-7 shadow-[0_30px_60px_-30px_rgba(20,18,16,0.35)] sm:px-8">
      <p className="mb-5 flex items-center gap-2 border-b border-ink/10 pb-3 font-mono text-xs text-ink-soft">
        <span className="h-2 w-2 rounded-full bg-shu" /># ask-sparky
      </p>
      <div className="space-y-4">
        <Appear progress={progress} i={0}>
          <div className="flex justify-end">
            <p className="rounded-2xl rounded-br-sm bg-ink px-4 py-2.5 text-sm text-paper">
              what are the prerequisites for CSE 485?
            </p>
          </div>
        </Appear>
        <Appear progress={progress} i={1}>
          <Trace icon={<BookOpen className="h-3.5 w-3.5" aria-hidden />} text="read 4 sources from the knowledge base" />
        </Appear>
        <Appear progress={progress} i={2}>
          <Trace icon={<Search className="h-3.5 w-3.5" aria-hidden />} text="search_live · ASU Course Catalog · CSE 485" />
        </Appear>
        <Appear progress={progress} i={3}>
          <Trace icon={<Terminal className="h-3.5 w-3.5" aria-hidden />} text="run_sandbox · opened the catalog entry" />
        </Appear>
        <Appear progress={progress} i={4}>
          <div className="space-y-3">
            <p className="border-l-4 border-kin pl-4 text-[0.95rem] leading-7">
              CSE 485 (Senior Project I) needs CSE 310 and CSE 340, and you have to be a CSE major with
              senior standing. It carries 3 credit hours and runs as a two-semester sequence with CSE 486.
            </p>
            <a
              href={CITE.href}
              target="_blank"
              rel="noreferrer"
              className="ml-5 inline-flex items-center gap-1.5 rounded-full bg-shu px-3 py-1 text-xs font-semibold text-paper"
            >
              {CITE.label}
              <ArrowUpRight className="h-3 w-3" aria-hidden />
            </a>
          </div>
        </Appear>
      </div>
    </div>
    <div className="relative z-10 mx-auto -mt-1 flex h-5 w-[108%] -translate-x-[3.7%] items-center justify-between rounded-full bg-ink px-1">
      <span className="h-6 w-6 rounded-full bg-kin" />
      <span className="h-6 w-6 rounded-full bg-kin" />
    </div>
  </div>
);

const Trace = ({ icon, text }: { icon: ReactNode; text: string }) => (
  <p className="flex items-center gap-2 font-mono text-xs text-ink-soft">
    <span className="text-shu">{icon}</span>
    {text}
  </p>
);

/** One turn, played back as the reader scrolls, beside the steps it takes. */
const InAction = () => {
  const ref = useRef<HTMLElement>(null);
  const pinned = usePinned();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const settled = useMotionValue(1);
  const progress = pinned ? scrollYProgress : settled;

  return (
    <section
      id="in-action"
      ref={ref}
      aria-label="Sparky in action"
      className={`relative z-10 ${pinned ? "h-[340vh]" : "py-10"}`}
    >
      <div className={pinned ? "sticky top-0 flex h-screen items-center" : ""}>
        <div className="mx-auto grid w-full max-w-6xl gap-12 px-5 sm:px-8 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div>
            <Heading
              numeral="壱"
              label="In action"
              title="It shows its working."
            />
            {pinned && (
              <ol className="mt-10 space-y-5">
                {STEPS.map((step, i) => (
                  <StepRow key={step.label} progress={progress} i={i} step={step} />
                ))}
              </ol>
            )}
          </div>
          <ScrollPaper progress={progress} />
        </div>
      </div>
    </section>
  );
};

export default InAction;
