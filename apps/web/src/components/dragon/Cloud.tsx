/** An auspicious cloud scroll, drawn as a gold outline. */
const Cloud = ({ className = "" }: { className?: string }) => (
  <svg aria-hidden viewBox="0 0 220 90" className={className} fill="none" strokeLinecap="round" strokeLinejoin="round">
    <path
      d="M10 78 H 200 C 214 78 214 58 200 58 C 206 40 186 28 172 38 C 168 18 140 12 128 30 C 118 8 82 8 76 34 C 62 22 38 30 42 50 C 22 46 10 60 22 68 C 12 68 6 76 10 78 Z"
      fill="var(--color-paper)"
      stroke="var(--color-kin)"
      strokeWidth="3"
    />
    <path
      d="M76 34 c 10 4 14 16 6 22 c -6 4 -14 -2 -10 -8 c 2 -3 6 -2 6 1 M128 30 c 10 2 16 14 8 22 c -6 4 -12 -2 -8 -6 M172 38 c 8 4 8 14 0 16"
      stroke="var(--color-kin)"
      strokeWidth="3"
    />
    <path d="M30 70 c 20 -6 40 -6 60 0 M110 70 c 20 -6 40 -6 60 0" stroke="var(--color-shu)" strokeWidth="2" opacity=".6" />
  </svg>
);

export default Cloud;
