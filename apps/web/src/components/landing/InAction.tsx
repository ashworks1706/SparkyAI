import { useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Check, ExternalLink, Terminal } from "lucide-react";
import { GlassPanel, Section, SectionLabel } from "./Glass";

type Step =
  | { kind: "ask"; text: string }
  | { kind: "trace"; icon: "read" | "run"; text: string }
  | { kind: "answer"; text: string; cite: { label: string; href: string } };

/** One turn, as the bot actually reports it: the steps it took, then the answer and its source. */
const TURN: Step[] = [
  { kind: "ask", text: "what are the prerequisites for CSE 485?" },
  {
    kind: "trace",
    icon: "read",
    text: "read 4 sources from the knowledge base",
  },
  {
    kind: "trace",
    icon: "run",
    text: "search_live · ASU Course Catalog · CSE 485",
  },
  {
    kind: "trace",
    icon: "run",
    text: "run_sandbox · opened the catalog entry",
  },
  {
    kind: "answer",
    text: "CSE 485 (Senior Project I) needs CSE 310 and CSE 340, and you have to be a CSE major with senior standing. It carries 3 credit hours and runs as a two-semester sequence with CSE 486.",
    cite: {
      label: "ASU Course Catalog",
      href: "https://catalog.apps.asu.edu/catalog/courses",
    },
  },
];

/**
 * Whether the turn can be revealed a step at a time: it needs an observer to know the block is on
 * screen, and a reader who asked for less motion is not shown a reveal at all.
 */
const canStagger = () =>
  typeof IntersectionObserver !== "undefined" &&
  !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Replays the turn once the block is on screen, a step at a time. */
const useReplay = (count: number) => {
  const ref = useRef<HTMLDivElement>(null);
  // Anyone who gets no reveal starts with the whole turn, rather than with an empty box.
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
    const timer = window.setTimeout(
      () => setShown((n) => n + 1),
      shown === 1 ? 500 : 750,
    );
    return () => window.clearTimeout(timer);
  }, [shown, count]);

  return { ref, shown };
};

const TraceLine = ({ step }: { step: Extract<Step, { kind: "trace" }> }) => (
  <p className="flex items-center gap-2 text-xs text-stone-500">
    {step.icon === "read" ? (
      <BookOpen
        className="h-3.5 w-3.5 shrink-0 text-sparky-maroon/70"
        aria-hidden
      />
    ) : (
      <Terminal
        className="h-3.5 w-3.5 shrink-0 text-sparky-maroon/70"
        aria-hidden
      />
    )}
    <span className="font-mono">{step.text}</span>
  </p>
);

const InAction = () => {
  const { ref, shown } = useReplay(TURN.length);
  const visible = useMemo(() => TURN.slice(0, shown), [shown]);
  const done = shown >= TURN.length;

  return (
    <Section id="in-action" label="Sparky in action">
      <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:items-center lg:gap-16">
        <div>
          <SectionLabel>In action</SectionLabel>
          <h2 className="mt-4 text-balance text-4xl font-semibold tracking-[-0.035em] text-stone-900 sm:text-5xl">
            It shows its working
          </h2>
          <p className="mt-5 text-pretty leading-7 text-stone-600">
            Every turn reports what it did: the stored pages it read, the source
            it fetched live, and the page it opened itself when a search came
            back thin. The answer arrives with the page behind it, so a student
            can check it in one click.
          </p>
          <ul className="mt-7 space-y-3">
            {[
              "A failed search is not an answer; it opens the page instead",
              "Nothing is quoted that it was not given this turn",
              "Every claim carries the source and the date it was read",
            ].map((line) => (
              <li
                key={line}
                className="flex items-start gap-3 text-sm text-stone-700"
              >
                <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-sparky-maroon/10">
                  <Check className="h-3 w-3 text-sparky-maroon" aria-hidden />
                </span>
                {line}
              </li>
            ))}
          </ul>
        </div>

        <GlassPanel className="p-2 sm:p-3">
          <div className="rounded-[1.35rem] bg-white/70 p-5 sm:p-6" ref={ref}>
            <div className="flex items-center gap-2 border-b border-stone-200/70 pb-4">
              <span className="h-2.5 w-2.5 rounded-full bg-sparky-maroon/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-stone-300" />
              <span className="h-2.5 w-2.5 rounded-full bg-stone-300" />
              <span className="ml-2 font-mono text-[0.7rem] text-stone-400">
                #ask-sparky
              </span>
            </div>

            <div className="mt-5 min-h-[20rem] space-y-4">
              {visible.map((step, i) =>
                step.kind === "ask" ? (
                  <div key={i} className="flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-md bg-sparky-maroon px-4 py-2.5 text-sm text-white shadow-maroon">
                      {step.text}
                    </p>
                  </div>
                ) : step.kind === "trace" ? (
                  <TraceLine key={i} step={step} />
                ) : (
                  <div
                    key={i}
                    className="space-y-3 motion-safe:animate-fade-up"
                  >
                    <p className="max-w-[92%] rounded-2xl rounded-bl-md border border-white/80 bg-white/90 px-4 py-3 text-sm leading-6 text-stone-700 shadow-glass">
                      {step.text}
                    </p>
                    <a
                      href={step.cite.href}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 rounded-full border border-white/80 bg-white/70 px-3 py-1.5 text-xs font-medium text-stone-700 shadow-glass transition-colors hover:text-sparky-maroon"
                    >
                      {step.cite.label}
                      <ExternalLink className="h-3 w-3" aria-hidden />
                    </a>
                  </div>
                ),
              )}
              {!done && shown > 0 && (
                <span className="inline-block h-4 w-1.5 bg-sparky-maroon/70 motion-safe:animate-caret" />
              )}
            </div>
          </div>
        </GlassPanel>
      </div>
    </Section>
  );
};

export default InAction;
