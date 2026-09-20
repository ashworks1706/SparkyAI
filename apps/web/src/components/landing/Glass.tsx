import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type GlassProps = {
  children: ReactNode;
  className?: string;
  /** Lifts the panel on hover. Use on anything the reader can click. */
  interactive?: boolean;
};

/**
 * A frosted panel. The white wash and the inset highlight are what read as glass over the
 * ambient gradient; the blur alone is not enough on a light background.
 */
export const GlassPanel = ({
  children,
  className,
  interactive = false,
}: GlassProps) => (
  <div
    className={cn(
      "relative overflow-hidden rounded-3xl border border-white/70 bg-white/55 shadow-glass backdrop-blur-xl",
      "before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px",
      "before:bg-gradient-to-r before:from-transparent before:via-white before:to-transparent",
      interactive &&
        "transition-all duration-300 hover:-translate-y-1 hover:border-white/90 hover:bg-white/70 hover:shadow-glass-lg",
      className,
    )}
  >
    {children}
  </div>
);

/** A small frosted pill, for labels and trust markers. */
export const GlassPill = ({
  children,
  className,
}: Omit<GlassProps, "interactive">) => (
  <span
    className={cn(
      "inline-flex items-center gap-2 rounded-full border border-white/70 bg-white/60 px-3.5 py-1.5",
      "text-xs font-medium text-stone-700 shadow-glass backdrop-blur-xl",
      className,
    )}
  >
    {children}
  </span>
);

/** The label above a section heading. */
export const SectionLabel = ({ children }: { children: ReactNode }) => (
  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-sparky-maroon">
    {children}
  </p>
);

type SectionProps = {
  children: ReactNode;
  className?: string;
  id?: string;
  /** Read out by screen readers as the name of the region. */
  label: string;
};

/** One band of the page, with the width and rhythm every section shares. */
export const Section = ({ children, className, id, label }: SectionProps) => (
  <section
    id={id}
    aria-label={label}
    className={cn("relative px-5 py-20 sm:px-8 sm:py-28 lg:px-10", className)}
  >
    <div className="mx-auto w-full max-w-6xl">{children}</div>
  </section>
);
