/** Head edge from the top of the neck, over the crest and snout, around the open jaw, to the throat. */
const EDGE =
  "M -40 -17 C -10 -18 14 -30 30 -52 L 44 -46 C 60 -44 78 -40 92 -32 C 100 -28 104 -24 108 -18 L 150 4 C 156 8 154 14 148 15 L 112 18 L 104 24 L 132 36 C 136 42 130 48 122 45 L 96 52 C 70 58 44 50 28 40 L 20 30 C 4 24 -14 17 -40 17";

/** The logo dragon head facing right, drawn around its neck at the origin. */
const DragonHead = () => (
  <g strokeLinejoin="round">
    <g fill="var(--color-enji-deep)">
      <path d="M 20 -44 C 0 -76 -28 -110 -58 -138 C -63 -143 -56 -148 -51 -143 C -20 -116 12 -82 36 -48 Z" />
      <path d="M 38 -46 C 28 -80 10 -116 -12 -146 C -16 -152 -8 -155 -5 -149 C 18 -120 40 -84 54 -44 Z" />
    </g>
    <path d={EDGE} fill="none" stroke="var(--color-enji-deep)" strokeWidth="28" />
    <path d={EDGE} fill="none" stroke="var(--color-kin)" strokeWidth="4" />
    <path d={`${EDGE} Z`} fill="var(--color-enji)" />
    <path
      d="M 68 -18 C 74 -27 88 -27 96 -18 C 88 -11 74 -11 68 -18 Z"
      fill="var(--color-enji)"
      stroke="var(--color-kin)"
      strokeWidth="3"
    />
  </g>
);

export default DragonHead;
