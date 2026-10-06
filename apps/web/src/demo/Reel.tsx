import type { Scene } from "./setup/kit";
import { Keys, Pointer, Stage } from "./Stage";
import { at, pressAt, progress, type Shot } from "./timeline";

/** Where every window sits on the stage. */
const FRAME = { x: 80, y: 70, w: 1440, h: 860 };

/** The chapter label in the corner: step number and title. */
const chapter = (i: number, of: number, title: string, opacity: number) => (
  <div
    className="absolute left-8 top-6 flex items-center gap-3 rounded-full bg-[#0d0d10]/85 py-2 pl-2 pr-5 font-sans text-white shadow-lg ring-1 ring-white/15"
    style={{ opacity }}
  >
    <span className="rounded-full bg-white/15 px-2.5 py-0.5 text-sm font-semibold tabular-nums">
      {i + 1}/{of}
    </span>
    <span className="text-lg font-medium">{title}</span>
  </div>
);

/** A video of scenes played one after another, each fading into the next under a chapter label. */
export const reel = (scenes: Scene[]) => {
  const starts = scenes.reduce<number[]>((s, _, i) => [...s, i === 0 ? 0 : s[i - 1] + scenes[i - 1].length], []);
  const length = Math.round(starts[starts.length - 1] + scenes[scenes.length - 1].length);
  const merged = (pick: (s: Scene) => Shot[]) => scenes.flatMap((s, i) => pick(s).map((stop) => ({ ...stop, t: stop.t + starts[i] })));
  const shots = merged((s) => s.shots);
  const pointer = merged((s) => s.pointer);
  const clicks = scenes.flatMap((s, i) => s.clicks.map((c) => c + starts[i]));
  const keys = scenes.flatMap((s, i) => s.keys.map((k) => ({ ...k, t: k.t + starts[i] })));
  const span = (i: number): [number, number] => [starts[i], i === scenes.length - 1 ? length + 1 : starts[i] + scenes[i].length];
  const shown = (t: number, [from, to]: [number, number]) =>
    Math.min(from === 0 ? 1 : progress(t, from - 0.15, 0.3), 1 - progress(t, to - 0.15, 0.3));

  /** One frame of the reel at time t. */
  const Frame = ({ t }: { t: number }) => {
    const key = keys.filter((k) => t >= k.t && t < k.t + 0.9).pop();
    const keyOpacity = key ? Math.min(progress(t, key.t, 0.12), 1 - progress(t, key.t + 0.7, 0.2)) : 0;
    const current = Math.max(0, starts.filter((s) => t >= s).length - 1);
    const here = at(pointer, t);
    return (
      <Stage
        shot={at(shots, t)}
        overlay={
          <>
            {chapter(current, scenes.length, scenes[current].title, shown(t, span(current)))}
            {key && <Keys keys={key.keys} opacity={keyOpacity} />}
          </>
        }
      >
        {scenes.map((scene, i) => {
          const opacity = shown(t, span(i));
          return opacity > 0 ? (
            <div
              key={scene.title}
              className="absolute"
              style={{ left: FRAME.x, top: FRAME.y, width: FRAME.w, height: FRAME.h, opacity, transform: `scale(${0.97 + 0.03 * opacity})` }}
            >
              {scene.view(t - starts[i])}
            </div>
          ) : null;
        })}
        <Pointer x={here.x} y={here.y} pressed={pressAt(clicks, t)} />
      </Stage>
    );
  };

  return { Frame, length };
};
