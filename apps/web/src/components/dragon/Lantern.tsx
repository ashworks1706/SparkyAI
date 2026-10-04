type LanternProps = {
  /** Length of the cord above the lantern, in pixels. */
  cord?: number;
  /** Width of the lantern body, in pixels. */
  size?: number;
  /** Seconds the sway is offset by, so a row does not move in step. */
  delay?: number;
  className?: string;
};

/** A red paper lantern on a cord, swaying from its top. */
const Lantern = ({ cord = 40, size = 56, delay = 0, className = "" }: LanternProps) => (
  <div
    aria-hidden
    className={`origin-top motion-safe:animate-[sway_4.5s_ease-in-out_infinite] ${className}`}
    style={{ animationDelay: `${-delay}s` }}
  >
    <div className="mx-auto w-px bg-ink/60" style={{ height: cord }} />
    <svg width={size} height={size * 1.55} viewBox="0 0 60 93" className="drop-shadow-[0_8px_18px_rgba(181,24,43,0.35)]">
      <rect x="20" y="0" width="20" height="7" rx="1.5" fill="var(--color-ink)" />
      <ellipse cx="30" cy="38" rx="28" ry="31" fill="var(--color-shu)" />
      <ellipse cx="30" cy="38" rx="18" ry="31" fill="none" stroke="var(--color-shu-deep)" strokeWidth="1.5" />
      <ellipse cx="30" cy="38" rx="7" ry="31" fill="none" stroke="var(--color-shu-deep)" strokeWidth="1.5" />
      <ellipse cx="24" cy="30" rx="10" ry="16" fill="#ff8f5a" opacity=".35" />
      <rect x="18" y="5" width="24" height="5" rx="1" fill="var(--color-kin)" />
      <rect x="18" y="66" width="24" height="5" rx="1" fill="var(--color-kin)" />
      <rect x="22" y="71" width="16" height="4" fill="var(--color-ink)" />
      <path d="M26 75 v16 M30 75 v18 M34 75 v16" stroke="var(--color-kin)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  </div>
);

export default Lantern;
