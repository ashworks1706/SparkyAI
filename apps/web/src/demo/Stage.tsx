import type { ReactNode } from "react";
import { STAGE, type Shot } from "./timeline";

/** The backdrop and the camera: children sit on the stage and the camera centres shot. */
export const Stage = ({ shot, children, overlay }: { shot: Shot; children: ReactNode; overlay?: ReactNode }) => (
  <div
    className="relative overflow-hidden"
    style={{
      width: STAGE.w,
      height: STAGE.h,
      background: "radial-gradient(120% 90% at 20% 0%, #9b2a48 0%, #781c35 35%, #3a0d1a 100%)",
    }}
  >
    <div
      className="absolute left-0 top-0 origin-top-left"
      style={{
        width: STAGE.w,
        height: STAGE.h,
        transform: `translate(${STAGE.w / 2 - shot.x * shot.s}px, ${STAGE.h / 2 - shot.y * shot.s}px) scale(${shot.s})`,
      }}
    >
      {children}
    </div>
    {overlay}
  </div>
);

/** The pointer Screen Studio style recordings show, scaled with the camera. */
export const Pointer = ({ x, y, pressed }: { x: number; y: number; pressed: number }) => (
  <div className="pointer-events-none absolute left-0 top-0" style={{ transform: `translate(${x}px, ${y}px)` }}>
    {pressed > 0 && (
      <span
        className="absolute -left-5 -top-5 h-10 w-10 rounded-full border-2 border-white/80"
        style={{ opacity: 1 - pressed, transform: `scale(${0.5 + pressed})` }}
      />
    )}
    <svg viewBox="0 0 24 24" width="30" height="30" style={{ transform: `scale(${pressed > 0 && pressed < 0.5 ? 0.85 : 1})` }}>
      <path d="M5 2.5v17.2l4.6-4.4 2.9 6.6 3-1.3-2.9-6.5h6.4L5 2.5Z" fill="#0b0b0c" stroke="white" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  </div>
);

/** A keystroke shown at the bottom of the frame, faded by opacity. */
export const Keys = ({ keys, opacity }: { keys: string[]; opacity: number }) =>
  opacity > 0 ? (
    <div className="absolute inset-x-0 bottom-10 flex justify-center gap-2" style={{ opacity }}>
      {keys.map((k) => (
        <span
          key={k}
          className="flex h-14 min-w-14 items-center justify-center rounded-xl bg-black/70 px-4 font-sans text-2xl font-semibold text-white shadow-lg ring-1 ring-white/15 backdrop-blur"
        >
          {k}
        </span>
      ))}
    </div>
  ) : null;
