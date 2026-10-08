/** A blinking caret, a terminal block unless className sizes it. */
export const Caret = ({ t, className = "h-6 w-2.5 translate-y-1 bg-[#c9d1d9]" }: { t: number; className?: string }) => (
  <span className={`inline-block ${className} ${Math.floor(t * 2.2) % 2 === 0 ? "" : "opacity-0"}`} />
);
