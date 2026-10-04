/** An eastern dragon head facing right, drawn around its neck at the origin. */
const DragonHead = ({ className = "" }: { className?: string }) => (
  <g className={className}>
    <g className="dragon-mane">
      <path
        d="M8 -34 C -10 -62 -42 -62 -66 -46 C -46 -46 -36 -38 -32 -30 C -56 -32 -74 -20 -84 0 C -62 -10 -46 -6 -38 2 C -60 8 -70 28 -68 50 C -52 32 -36 28 -24 30 C -32 46 -28 64 -16 74 C -14 54 0 42 16 36 Z"
        fill="var(--color-kin)"
        stroke="var(--color-shu-deep)"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <path
        d="M6 -26 C -12 -42 -32 -40 -46 -32 C -32 -28 -26 -22 -24 -16 C -40 -14 -50 -4 -52 10 C -40 4 -28 6 -20 12 C -30 26 -28 40 -20 48 C -14 36 -2 28 10 26 Z"
        fill="var(--color-shu)"
      />
    </g>

    <g fill="var(--color-kin)" stroke="var(--color-shu-deep)" strokeWidth="2.5" strokeLinejoin="round">
      <path d="M42 -32 C 30 -58 4 -76 -34 -86 C -8 -68 8 -52 18 -30 Z" />
      <path d="M8 -66 C 0 -78 2 -90 10 -100 C 12 -86 18 -78 24 -68 Z" />
      <path d="M58 -34 C 56 -54 42 -70 18 -82 C 32 -66 38 -50 40 -32 Z" />
    </g>

    <path
      d="M12 20 C 40 32 82 42 122 36 C 128 42 124 52 114 54 C 78 60 36 52 6 36 Z"
      fill="var(--color-shu-deep)"
      stroke="var(--color-ink)"
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <g fill="var(--color-paper)" stroke="var(--color-ink)" strokeWidth="1">
      <path d="M44 33 l4 -9 l4 10 Z" />
      <path d="M68 38 l4 -10 l4 11 Z" />
      <path d="M92 40 l4 -10 l4 10 Z" />
      <path d="M110 39 l3 -10 l5 8 Z" />
    </g>
    <path d="M60 42 C 84 46 104 40 118 44 C 112 48 102 50 96 47" fill="none" stroke="#e8435a" strokeWidth="4" strokeLinecap="round" />

    <g fill="var(--color-kin)" stroke="var(--color-shu-deep)" strokeWidth="2" strokeLinejoin="round">
      <path d="M30 46 C 34 64 26 80 12 92 C 30 86 40 76 44 64 C 48 78 44 92 36 102 C 54 92 60 76 58 56 Z" />
    </g>

    <path
      d="M-4 -26 C 14 -42 44 -44 60 -36 C 70 -48 90 -46 98 -30 C 116 -26 134 -24 148 -20 C 160 -24 168 -14 162 -4 C 156 6 140 8 128 12 C 108 20 86 22 64 24 C 40 26 16 24 -4 28 C -14 10 -14 -10 -4 -26 Z"
      fill="var(--color-shu)"
      stroke="var(--color-ink)"
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
    <path d="M10 -4 C 40 -10 70 -6 96 0 M14 12 C 44 16 82 16 120 10" fill="none" stroke="var(--color-kin)" strokeWidth="2.5" strokeLinecap="round" />
    <path d="M152 -14 c 6 -2 8 6 2 8 c -4 1 -6 -3 -3 -5" fill="none" stroke="var(--color-ink)" strokeWidth="2" strokeLinecap="round" />
    <path
      d="M54 -34 C 66 -56 94 -58 106 -34 L 96 -36 C 88 -46 72 -46 64 -34 Z"
      fill="var(--color-kin)"
      stroke="var(--color-shu-deep)"
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <path d="M66 -22 C 72 -32 90 -32 98 -22 C 90 -16 76 -15 66 -22 Z" fill="var(--color-kin)" stroke="var(--color-ink)" strokeWidth="2" />
    <circle cx="84" cy="-23" r="4.5" fill="var(--color-ink)" />
    <circle cx="85.5" cy="-24.5" r="1.4" fill="var(--color-paper)" />

    <g className="dragon-whiskers" fill="none" stroke="var(--color-kin)" strokeLinecap="round">
      <path d="M158 2 C 172 34 140 66 90 70 C 44 74 6 96 -26 128" strokeWidth="3" />
      <path d="M150 -20 C 170 -50 150 -82 110 -86 C 70 -90 30 -112 -10 -134" strokeWidth="2.5" />
    </g>
  </g>
);

export default DragonHead;
