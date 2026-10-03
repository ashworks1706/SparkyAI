/** A red seal stamp carrying one character. */
const Hanko = ({ char = "龍", className = "" }: { char?: string; className?: string }) => (
  <span
    aria-hidden
    className={`grid aspect-square place-items-center rounded-[3px] bg-shu font-serif font-bold leading-none text-paper shadow-[inset_0_0_0_2px_var(--color-shu),inset_0_0_0_3.5px_var(--color-paper)] ${className}`}
  >
    {char}
  </span>
);

export default Hanko;
