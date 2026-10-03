import Section from "./Section";

type Source = { title: string; ask: string; answer: string; source: string };

/** What students ask, and the source that answers each one. */
const SOURCES: Source[] = [
  {
    title: "Courses",
    ask: "what do I need before CSE 310?",
    answer: "The catalog entry, its credit hours and prerequisites, then open sections for the term.",
    source: "Class Search, Course Catalog",
  },
  {
    title: "Scholarships",
    ask: "scholarships for a CS junior",
    answer: "What is open, who may apply and when it closes, off the scholarship portal itself.",
    source: "Scholarship Search",
  },
  {
    title: "Library hours",
    ask: "when does Hayden close tonight?",
    answer: "This week's hours for every library, with the day you asked for read off the row.",
    source: "ASU Library",
  },
  {
    title: "Shuttles",
    ask: "next shuttle to Poly",
    answer: "Live departures from the tracker. Never a stored copy, because a stored copy is wrong.",
    source: "Shuttle Tracker",
  },
  {
    title: "Study rooms",
    ask: "a room for four at 6pm",
    answer: "Bookable slots on the date you name, and the link that books one.",
    source: "Library study rooms",
  },
  {
    title: "Dining",
    ask: "what's open on Tempe right now?",
    answer: "Hours per venue for the campus you name, off the current term's schedule.",
    source: "Sun Devil Dining",
  },
  {
    title: "Clubs",
    ask: "is there an AI club?",
    answer: "Organisations on Sun Devil Central, searched by the acronym you wrote.",
    source: "Sun Devil Central",
  },
  {
    title: "Events",
    ask: "anything on this weekend?",
    answer: "The events calendar, plus registrar dates for drop, add and graduation.",
    source: "Events, Registrar",
  },
];

/** The sources Sparky reaches, as a hairline grid of example questions. */
const Sources = () => (
  <Section
    id="sources"
    label="What students ask"
    numeral="二"
    margin="用途"
    title="One place for the things you would otherwise go digging for."
    lead="Each is a source Sparky knows how to reach, with the parameters it takes. Ask in plain words; it picks the source and writes the query."
  >
    <ul className="grid gap-px overflow-hidden rounded-xl border border-ink/10 bg-ink/10 sm:grid-cols-2 lg:grid-cols-4">
      {SOURCES.map((item) => (
        <li key={item.title} className="group flex flex-col bg-paper p-6 transition-colors hover:bg-white">
          <h3 className="flex items-center gap-2 font-serif text-lg font-semibold">
            <span className="h-1.5 w-1.5 rounded-full bg-shu transition-colors group-hover:bg-kin" />
            {item.title}
          </h3>
          <p className="mt-4 font-mono text-xs text-shu">&ldquo;{item.ask}&rdquo;</p>
          <p className="mt-3 flex-1 text-sm leading-6 text-ink-soft">{item.answer}</p>
          <p className="mt-5 text-[0.68rem] font-medium uppercase tracking-[0.15em] text-ink/45">
            {item.source}
          </p>
        </li>
      ))}
    </ul>
  </Section>
);

export default Sources;
