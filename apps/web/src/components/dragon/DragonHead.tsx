/** Head edge after the logo: crest, brow, snout, open jaw and throat, with the neck at the origin. */
const EDGE =
  "M -30 -22 C -6 -24 6 -30 14 -40 L 22 -26 C 46 -26 72 -22 92 -14 L 132 0 C 137 2 139 4 139 8 L 140 14 L 106 16 L 98 20 L 128 26 L 128 32 C 128 35 126 36 122 36 L 64 38 L 32 32 L 24 26 C 6 23 -12 22 -30 22";

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
        <path d="M 2 -30 C -8 -46 -20 -60 -34 -72 C -39 -77 -32 -81 -27 -76 C -12 -64 4 -50 16 -36 Z" />
        <path d="M 18 -30 C 14 -48 6 -64 -4 -78 C -7 -84 1 -87 4 -81 C 14 -66 24 -50 30 -28 Z" />
      </g>
      <path d="M 98 20 L 106 16 L 140 14 L 128 26 Z" fill="var(--color-shu-deep)" />
      <g fill="var(--color-paper)">
        <path d="M 114 15.6 l 2.5 5 l 2.5 -5 Z" />
        <path d="M 126 15 l 2.5 5 l 2.5 -5 Z" />
        <path d="M 110 23.5 l 2.5 -4 l 2.5 5 Z" />
      </g>
      <path d={`${EDGE} Z`} fill="var(--color-shu)" />
      <g clipPath="url(#dragon-head-edge)" fill="none">
        <path d={EDGE} stroke="var(--color-kin)" strokeWidth="9" />
        <path d={EDGE} stroke="var(--color-shu)" strokeWidth="5" />
      </g>
      <path d={EDGE} fill="none" stroke="var(--color-ink)" strokeWidth="1.6" />
      <path
        d="M 50 -8 C 56 -15 68 -15 74 -8 C 68 -3 56 -3 50 -8 Z"
        fill="var(--color-kin)"
        stroke="var(--color-ink)"
        strokeWidth="1.4"
      />
      <path d="M 98 20 C 80 22 62 24 44 28" fill="none" stroke="var(--color-kin)" strokeWidth="2" strokeLinecap="round" />
      <path d="M 128 3 c 3 -2 6 1 4 4" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" strokeLinecap="round" />
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
