/** Head edge after the logo: crest, brow, snout, open jaw and throat, with the neck at the origin. */
const EDGE =
  "M -30 -22 C 4 -24 20 -36 30 -52 L 44 -46 C 60 -44 78 -40 92 -32 C 100 -28 104 -24 108 -18 L 150 4 C 156 8 154 14 148 15 L 112 18 L 104 24 L 132 36 C 136 42 130 48 122 45 L 96 52 C 70 58 44 50 28 40 L 20 30 C 4 26 -14 22 -30 22";

/** An eastern dragon head with the logo's face, facing right, drawn around its neck at the origin. */
const DragonHead = ({ className = "" }: { className?: string }) => (
  <g className={className}>
    <g transform="scale(1.6)" strokeLinejoin="round">
      <defs>
        <clipPath id="dragon-head-edge">
          <path d={`${EDGE} Z`} />
        </clipPath>
      </defs>
      <g fill="var(--color-kin)" stroke="var(--color-shu-deep)" strokeWidth="1.5">
        <path d="M 20 -44 C 0 -76 -28 -110 -58 -138 C -63 -143 -56 -148 -51 -143 C -20 -116 12 -82 36 -48 Z" />
        <path d="M 38 -46 C 28 -80 10 -116 -12 -146 C -16 -152 -8 -155 -5 -149 C 18 -120 40 -84 54 -44 Z" />
      </g>
      <path d="M 104 24 L 112 18 L 148 15 L 132 36 Z" fill="var(--color-shu-deep)" />
      <g fill="var(--color-paper)">
        <path d="M 118 18 l 3 6 l 3 -6 Z" />
        <path d="M 130 17 l 3 6 l 3 -6 Z" />
        <path d="M 112 27 l 3 -5 l 3 6 Z" />
      </g>
      <path d={`${EDGE} Z`} fill="var(--color-shu)" />
      <g clipPath="url(#dragon-head-edge)" fill="none">
        <path d={EDGE} stroke="var(--color-kin)" strokeWidth="9" />
        <path d={EDGE} stroke="var(--color-shu)" strokeWidth="5" />
      </g>
      <path d={EDGE} fill="none" stroke="var(--color-ink)" strokeWidth="1.6" />
      <path
        d="M 68 -18 C 74 -27 88 -27 96 -18 C 88 -11 74 -11 68 -18 Z"
        fill="var(--color-kin)"
        stroke="var(--color-ink)"
        strokeWidth="1.4"
      />
      <circle cx="83" cy="-18" r="3" fill="var(--color-ink)" />
      <path d="M 140 4 c 3 -2 6 1 4 4" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" strokeLinecap="round" />
    </g>

    <g className="dragon-mane" transform="translate(-14 0)">
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
  </g>
);

export default DragonHead;
