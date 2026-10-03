import type { ReactNode } from "react";

type SectionProps = {
  id: string;
  /** Accessible name of the region. */
  label: string;
  /** Kanji numeral shown beside the heading. */
  numeral: string;
  /** Japanese word set vertically in the margin. */
  margin: string;
  title: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
};

/** A page section: numbered header, vertical margin word, then content. */
const Section = ({ id, label, numeral, margin, title, lead, children }: SectionProps) => (
  <section
    id={id}
    aria-label={label}
    className="relative scroll-mt-20 border-t border-ink/10 px-5 py-20 sm:px-8 sm:py-28"
  >
    <div className="mx-auto max-w-6xl">
      <div className="grid gap-6 lg:grid-cols-[4rem_1fr] lg:gap-10">
        <div className="hidden lg:flex lg:flex-col lg:items-center lg:gap-4">
          <span className="font-serif text-3xl font-semibold text-shu">{numeral}</span>
          <span className="h-10 w-px bg-ink/15" />
          <span className="tategaki font-serif text-sm tracking-[0.4em] text-ink-soft">
            {margin}
          </span>
        </div>
        <div>
          <p className="flex items-center gap-3 text-xs font-medium uppercase tracking-[0.25em] text-ink-soft">
            <span className="font-serif text-base text-shu lg:hidden">{numeral}</span>
            <span className="h-px w-6 bg-shu" />
            {label}
          </p>
          <h2 className="mt-5 max-w-3xl text-balance font-serif text-3xl font-semibold leading-tight tracking-tight sm:text-[2.6rem]">
            {title}
          </h2>
          {lead && (
            <p className="mt-5 max-w-2xl text-pretty leading-7 text-ink-soft">{lead}</p>
          )}
          <div className="mt-12">{children}</div>
        </div>
      </div>
    </div>
  </section>
);

export default Section;
