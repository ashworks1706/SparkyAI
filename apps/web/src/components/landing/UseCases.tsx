import type { LucideIcon } from "lucide-react";
import {
  Bus,
  CalendarDays,
  Clock,
  GraduationCap,
  Library,
  Trophy,
  UtensilsCrossed,
  Users,
} from "lucide-react";
import { GlassPanel, Section, SectionLabel } from "./Glass";

type UseCase = {
  icon: LucideIcon;
  title: string;
  ask: string;
  answer: string;
  source: string;
  /** Rows the card spans on a wide screen. The first two carry more, so they run wider. */
  wide?: boolean;
};

/** What students actually ask, and the source that answers each one. */
const CASES: UseCase[] = [
  {
    icon: GraduationCap,
    title: "Courses and prerequisites",
    ask: "what do I need before CSE 310?",
    answer:
      "The catalog entry, its credit hours and the prerequisites it lists, then the open sections for the term you asked about.",
    source: "ASU Class Search and Course Catalog",
    wide: true,
  },
  {
    icon: Trophy,
    title: "Scholarships",
    ask: "scholarships for a CS junior",
    answer:
      "What is open, who may apply and when it closes, read off the scholarship portal rather than a page about it.",
    source: "ASU Scholarship Search",
    wide: true,
  },
  {
    icon: Clock,
    title: "Library hours",
    ask: "when does Hayden close tonight?",
    answer:
      "This week's hours for every library, with the day you asked for read off the row.",
    source: "ASU Library",
  },
  {
    icon: Bus,
    title: "Shuttles",
    ask: "next shuttle to Poly",
    answer:
      "Live departures from the tracker. Never a stored copy, because a stored copy is wrong.",
    source: "ASU Shuttle Tracker",
  },
  {
    icon: Library,
    title: "Study rooms",
    ask: "a room for four at 6pm",
    answer: "Bookable slots on the date you name, and the link that books one.",
    source: "ASU Library study rooms",
  },
  {
    icon: UtensilsCrossed,
    title: "Dining",
    ask: "what's open on Tempe right now?",
    answer:
      "Hours per venue for the campus you name, off the current term's schedule.",
    source: "Sun Devil Dining",
  },
  {
    icon: Users,
    title: "Clubs",
    ask: "is there an AI club?",
    answer:
      "Organisations on Sun Devil Central. It searches the acronym you wrote rather than guessing what it stands for.",
    source: "Sun Devil Central",
  },
  {
    icon: CalendarDays,
    title: "Events and deadlines",
    ask: "anything on this weekend?",
    answer:
      "The events calendar, plus registrar dates for drop, add and graduation.",
    source: "ASU Events and Registrar",
  },
];

const Card = ({ item }: { item: UseCase }) => {
  const Icon = item.icon;
  return (
    <GlassPanel
      interactive
      className={`group flex flex-col p-6 ${item.wide ? "sm:col-span-2 lg:col-span-3" : "lg:col-span-2"}`}
    >
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-sparky-maroon/10 text-sparky-maroon transition-colors group-hover:bg-sparky-maroon group-hover:text-white">
          <Icon className="h-5 w-5" aria-hidden />
        </span>
        <h3 className="text-base font-semibold tracking-[-0.01em] text-stone-900">
          {item.title}
        </h3>
      </div>

      <p className="mt-5 rounded-xl rounded-bl-sm bg-stone-900/[0.04] px-3.5 py-2.5 font-mono text-xs text-stone-600">
        &ldquo;{item.ask}&rdquo;
      </p>
      <p className="mt-3 flex-1 text-sm leading-6 text-stone-600">
        {item.answer}
      </p>
      <p className="mt-4 border-t border-stone-200/70 pt-3 text-[0.7rem] font-medium uppercase tracking-[0.12em] text-stone-400">
        {item.source}
      </p>
    </GlassPanel>
  );
};

const UseCases = () => (
  <Section id="use-cases" label="What students ask">
    <div className="max-w-2xl">
      <SectionLabel>Use cases</SectionLabel>
      <h2 className="mt-4 text-balance text-4xl font-semibold tracking-[-0.035em] text-stone-900 sm:text-5xl">
        One place for the things you would otherwise go digging for
      </h2>
      <p className="mt-5 text-pretty leading-7 text-stone-600">
        Each of these is a source Sparky knows how to reach, with the parameters
        it takes. Ask in plain words; it picks the source and writes the query.
      </p>
    </div>

    <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
      {CASES.map((item) => (
        <Card key={item.title} item={item} />
      ))}
    </div>
  </Section>
);

export default UseCases;
