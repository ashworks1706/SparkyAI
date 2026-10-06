import { LENGTH, SCENES, STARTS } from "./setup/reel";
import type { Scene } from "./setup/kit";
import { Keys, Pointer, Stage } from "./Stage";
import { at, pressAt, progress, type Shot } from "./timeline";

/** Where every window sits on the stage. */
const FRAME = { x: 80, y: 70, w: 1440, h: 860 };

/** Stops of every scene on one timeline. */
const merged = (pick: (s: Scene) => Shot[]) => SCENES.flatMap((s, i) => pick(s).map((stop) => ({ ...stop, t: stop.t + STARTS[i] })));

const SHOTS = merged((s) => s.shots);
const POINTER = merged((s) => s.pointer);
const CLICKS = SCENES.flatMap((s, i) => s.clicks.map((c) => c + STARTS[i]));
const KEYS = SCENES.flatMap((s, i) => s.keys.map((k) => ({ ...k, t: k.t + STARTS[i] })));

/** The window of scene i: it fades in at its start and out at its end; the last one holds. */
const spanOf = (i: number): [number, number] => [STARTS[i], i === SCENES.length - 1 ? LENGTH + 1 : STARTS[i] + SCENES[i].length];

/** How visible a scene is at time t. */
const shown = (t: number, [from, to]: [number, number]) =>
  Math.min(from === 0 ? 1 : progress(t, from - 0.15, 0.3), 1 - progress(t, to - 0.15, 0.3));

/** The chapter label in the corner: step number and title. */
const Chapter = ({ i, opacity }: { i: number; opacity: number }) => (
  <div
    className="absolute left-8 top-6 flex items-center gap-3 rounded-full bg-[#0d0d10]/85 py-2 pl-2 pr-5 font-sans text-white shadow-lg ring-1 ring-white/15"
    style={{ opacity }}
  >
    <span className="rounded-full bg-white/15 px-2.5 py-0.5 text-sm font-semibold tabular-nums">
      {i + 1}/{SCENES.length}
    </span>
    <span className="text-lg font-medium">{SCENES[i].title}</span>
  </div>
);

/** One frame of the setup video at time t. */
export const Frame = ({ t }: { t: number }) => {
  const shot = at(SHOTS, t);
  const pointer = at(POINTER, t);
  const key = KEYS.filter((k) => t >= k.t && t < k.t + 0.9).pop();
  const keyOpacity = key ? Math.min(progress(t, key.t, 0.12), 1 - progress(t, key.t + 0.7, 0.2)) : 0;
  const current = Math.max(0, STARTS.filter((s) => t >= s).length - 1);
  const chapter = shown(t, spanOf(current));
  return (
    <Stage
      shot={shot}
      overlay={
        <>
          <Chapter i={current} opacity={chapter} />
          {key && <Keys keys={key.keys} opacity={keyOpacity} />}
        </>
      }
    >
      {SCENES.map((scene, i) => {
        const opacity = shown(t, spanOf(i));
        return opacity > 0 ? (
          <div
            key={scene.title}
            className="absolute"
            style={{ left: FRAME.x, top: FRAME.y, width: FRAME.w, height: FRAME.h, opacity, transform: `scale(${0.97 + 0.03 * opacity})` }}
          >
            {scene.view(t - STARTS[i])}
          </div>
        ) : null;
      })}
      <Pointer x={pointer.x} y={pointer.y} pressed={pressAt(CLICKS, t)} />
    </Stage>
  );
};
