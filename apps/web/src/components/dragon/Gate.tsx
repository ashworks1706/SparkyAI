/** The top of a red ceremonial gate: capped top beam, tie beam and name plaque. */
const Gate = ({ className = "" }: { className?: string }) => (
  <svg aria-hidden viewBox="0 0 800 170" className={className}>
    <path d="M6 30 C 200 62 600 62 794 30 L 786 58 C 600 86 200 86 14 58 Z" fill="var(--color-ink)" />
    <path d="M40 62 C 220 88 580 88 760 62 L 756 90 C 580 110 220 110 44 90 Z" fill="var(--color-shu)" />
    <rect x="70" y="140" width="660" height="22" fill="var(--color-shu)" />
    <rect x="128" y="90" width="34" height="80" fill="var(--color-shu)" />
    <rect x="638" y="90" width="34" height="80" fill="var(--color-shu)" />
    <rect x="368" y="92" width="64" height="48" fill="var(--color-shu)" />
    <rect x="376" y="99" width="48" height="34" fill="var(--color-ink)" />
    <rect x="382" y="105" width="36" height="22" fill="none" stroke="var(--color-kin)" strokeWidth="2" />
    <circle cx="400" cy="116" r="5" fill="var(--color-kin)" />
  </svg>
);

export default Gate;
