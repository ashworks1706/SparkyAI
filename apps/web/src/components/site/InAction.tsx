import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, BookOpen, Terminal } from "lucide-react";
import Section from "./Section";

type Step =
  | { kind: "ask"; text: string }
  | { kind: "trace"; icon: "read" | "run"; text: string }
  | { kind: "answer"; text: string; cite: { label: string; href: string } };

/** One turn as the bot reports it: the steps it took, then the answer and its source. */
const TURN: Step[] = [
  { kind: "ask", text: "what are the prerequisites for CSE 485?" },
  { kind: "trace", icon: "read", text: "read 4 sources from the knowledge base" },
  { kind: "trace", icon: "run", text: "search_live · ASU Course Catalog · CSE 485" },
  { kind: "trace", icon: "run", text: "run_sandbox · opened the catalog entry" },
  {
    kind: "answer",
    text: "CSE 485 (Senior Project I) needs CSE 310 and CSE 340, and you have to be a CSE major with senior standing. It carries 3 credit hours and runs as a two-semester sequence with CSE 486.",
    cite: {
      label: "ASU Course Catalog",
      href: "https://catalog.apps.asu.edu/catalog/courses",
    },
  },
];

const PRINCIPLES = [
  "A failed search is not an answer; it opens the page instead.",
  "Nothing is quoted that it was not given this turn.",
  "Every claim carries the source and the date it was read.",
];

/** Whether the turn is revealed step by step: needs an observer and no reduced-motion preference. */
const canStagger = () =>
  typeof IntersectionObserver !== "undefined" &&
  !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Replays the turn a step at a time once the block is on screen. */
const useReplay = (count: number) => {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(() => (canStagger() ? 0 : count));

  useEffect(() => {
    const node = ref.current;
    if (!node || !canStagger()) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setShown(1);
          observer.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [count]);

  useEffect(() => {
    if (shown === 0 || shown >= count) return;
    const timer = window.setTimeout(() => setShown((n) => n + 1), shown === 1 ? 500 : 750);
    return () => window.clearTimeout(timer);
  }, [shown, count]);

  return { ref, shown };
};

const Turn = ({ step }: { step: Step }) => {
  if (step.kind === "ask") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink px-4 py-2.5 text-sm text-paper">
          {step.text}
        </p>
      </div>
    );
  }
  if (step.kind === "trace") {
    const Icon = step.icon === "read" ? BookOpen : Terminal;
    return (
      <p className="flex items-center gap-2 text-xs text-ink-soft">
        <Icon className="h-3.5 w-3.5 shrink-0 text-shu" aria-hidden />
        <span className="font-mono">{step.text}</span>
      </p>
    );
  }
  return (
    <div className="space-y-3 motion-safe:animate-rise">
      <p className="max-w-[94%] border-l-2 border-shu pl-4 text-sm leading-6 text-ink">
        {step.text}
      </p>
      <a
        href={step.cite.href}
        target="_blank"
        rel="noreferrer"
        className="ml-4 inline-flex items-center gap-1.5 rounded-full bg-kin/25 px-3 py-1 text-xs font-medium text-ink transition-colors hover:bg-kin"
      >
        {step.cite.label}
        <ArrowUpRight className="h-3 w-3" aria-hidden />
      </a>
    </div>
  );
};

/** A replayed Discord turn beside the rules it follows. */
const InAction = () => {
  const { ref, shown } = useReplay(TURN.length);
  const done = shown >= TURN.length;

  return (
    <Section
      id="in-action"
      label="Sparky in action"
      numeral="一"
      margin="実演"
      title="It shows its working."
      lead="Every turn reports what it did: the stored pages it read, the source it fetched live, and the page it opened itself when a search came back thin. The answer arrives with the page behind it, so a student can check it in one click."
    >
      <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr] lg:items-start">
        <ol className="space-y-6">
          {PRINCIPLES.map((line, i) => (
            <li key={line} className="flex gap-4">
              <span className="font-serif text-sm font-semibold text-shu">0{i + 1}</span>
              <span className="text-pretty leading-7">{line}</span>
            </li>
          ))}
        </ol>

        <div ref={ref} className="rounded-xl border border-ink/15 bg-white">
          <div className="flex items-center justify-between border-b border-ink/10 px-5 py-3">
            <span className="font-mono text-xs text-ink-soft">#ask-sparky</span>
            <span className="flex gap-1.5">
              <span className="h-2 w-2 rounded-full bg-shu" />
              <span className="h-2 w-2 rounded-full bg-kin" />
              <span className="h-2 w-2 rounded-full bg-ink/20" />
            </span>
          </div>
          <div className="min-h-[21rem] space-y-4 p-5 sm:p-6">
            {TURN.slice(0, shown).map((step, i) => (
              <Turn key={i} step={step} />
            ))}
            {!done && shown > 0 && (
              <span className="inline-block h-4 w-1.5 bg-shu motion-safe:animate-caret" />
            )}
          </div>
        </div>
      </div>
    </Section>
  );
};

export default InAction;
