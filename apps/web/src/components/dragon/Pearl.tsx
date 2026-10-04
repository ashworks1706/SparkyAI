/** The flaming pearl the dragon chases, centered on the origin. */
const Pearl = () => (
  <g>
    <defs>
      <radialGradient id="pearl-glow" r="0.5">
        <stop offset="0" stopColor="#fff6cf" />
        <stop offset="0.55" stopColor="var(--color-kin)" />
        <stop offset="1" stopColor="var(--color-kin)" stopOpacity="0" />
      </radialGradient>
      <radialGradient id="pearl-body" cx="0.35" cy="0.35" r="0.7">
        <stop offset="0" stopColor="#fffdf2" />
        <stop offset="0.5" stopColor="#ffe48a" />
        <stop offset="1" stopColor="#e09a00" />
      </radialGradient>
    </defs>
    <circle r="46" fill="url(#pearl-glow)" opacity=".55">
      <animate attributeName="r" values="40;50;40" dur="2.4s" repeatCount="indefinite" />
    </circle>
    <g>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="9s" repeatCount="indefinite" />
      {[0, 72, 144, 216, 288].map((a) => (
        <path
          key={a}
          d="M0 -18 C 8 -26 6 -36 14 -42 C 10 -30 20 -26 12 -16 Z"
          fill="var(--color-shu)"
          transform={`rotate(${a})`}
        />
      ))}
    </g>
    <circle r="17" fill="url(#pearl-body)" stroke="var(--color-shu-deep)" strokeWidth="1.5" />
    <path d="M-8 -6 a 10 10 0 0 1 9 -6" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
  </g>
);

export default Pearl;
