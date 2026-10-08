import type { ReactNode } from "react";
import { typed, type Shot } from "../timeline";
import type { Code, Tone } from "./text";

/** A macOS window: traffic lights, a title, and its body. */
export const MacWindow = ({ title, dark = true, children }: { title: ReactNode; dark?: boolean; children: ReactNode }) => (
  <div
    className={`flex h-full w-full flex-col overflow-hidden rounded-xl shadow-[0_40px_80px_-30px_rgba(0,0,0,0.6)] ring-1 ${dark ? "bg-[#16161a] ring-black/50" : "bg-white ring-black/20"}`}
  >
    <div className={`relative flex h-11 shrink-0 items-center px-4 ${dark ? "bg-[#26262b] text-[#9c9ca6]" : "bg-[#ececee] text-[#55555c]"}`}>
      <span className="flex gap-2">
        <span className="h-3.5 w-3.5 rounded-full bg-[#ff5f57]" />
        <span className="h-3.5 w-3.5 rounded-full bg-[#febc2e]" />
        <span className="h-3.5 w-3.5 rounded-full bg-[#28c840]" />
      </span>
      <div className="absolute inset-x-0 flex justify-center text-sm font-medium">{title}</div>
    </div>
    <div className="min-h-0 flex-1">{children}</div>
  </div>
);

/** A blinking block caret. */
export const Caret = ({ t, className = "h-6 w-2.5 translate-y-1 bg-[#c9d1d9]" }: { t: number; className?: string }) => (
  <span className={`inline-block ${className} ${Math.floor(t * 2.2) % 2 === 0 ? "" : "opacity-0"}`} />
);

/** A shell prompt, as zsh shows it in the repository. */
export const Prompt = ({ dir }: { dir: string }) => (
  <>
    <span className="text-[#7ee787]">dev@laptop</span> <span className="text-[#79c0ff]">{dir}</span> <span className="text-[#c9d1d9]">$</span>{" "}
  </>
);

/** One line in a terminal: a command typed between at and to, or output printed at at. */
export type Line = { at: number; to?: number; text: string; tone?: "ok" | "dim" };

/** A terminal that plays lines in time, with a caret on the line being typed. */
export const Terminal = ({ lines, t, dir }: { lines: Line[]; t: number; dir: (i: number) => string }) => {
  const shown = lines.filter((l) => l.at <= t);
  const typing = shown.findIndex((l) => l.to !== undefined && t < l.to);
  const last = shown[shown.length - 1];
  const idle = typing === -1 && (!last || last.to === undefined || t >= last.to);
  return (
    <MacWindow title="~/SparkyAI — zsh">
      <div className="space-y-1.5 p-6 font-mono text-[19px] leading-7 text-[#c9d1d9]">
        {shown.map((l, i) =>
          l.to !== undefined ? (
            <p key={l.text}>
              <Prompt dir={dir(i)} />
              {typed(l.text, t, l.at, l.to)}
              {i === typing && <Caret t={t} />}
            </p>
          ) : (
            <p key={l.text} className={l.tone === "ok" ? "text-[#7ee787]" : l.tone === "dim" ? "text-[#8b949e]" : ""}>
              {l.text}
            </p>
          ),
        )}
        {idle && (
          <p>
            <Prompt dir="~/SparkyAI" />
            <Caret t={t} />
          </p>
        )}
      </div>
    </MacWindow>
  );
};

/** Syntax colours of the editor theme. */
const TONES: Record<Tone, string> = {
  key: "text-[#9cdcfe]",
  str: "text-[#ce9178]",
  num: "text-[#b5cea8]",
  com: "text-[#6a9955]",
  kw: "text-[#c586c0]",
  fn: "text-[#dcdcaa]",
  ty: "text-[#4ec9b0]",
};

/** A code editor with the explorer, tabs, and the open file. Added lines get the diff gutter. */
export const Editor = ({
  explorer,
  tabs,
  open,
  lines,
  added = [],
  cursor,
  t,
}: {
  explorer: string[];
  tabs: string[];
  open: string;
  lines: Code[];
  added?: number[];
  cursor?: number;
  t: number;
}) => (
  <MacWindow title="SparkyAI">
    <div className="flex h-full bg-[#1e1e1e] font-mono text-[16px] leading-[25px] text-[#d4d4d4]">
      <aside className="w-72 shrink-0 bg-[#181818] px-3 py-4 font-sans text-[14px] leading-6 text-[#cccccc]">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[#8b8b8b]">Explorer</p>
        {explorer.map((row) => (
          <p key={row} className={`whitespace-pre rounded px-2 ${row.trim() === open ? "bg-[#37373d] text-white" : ""}`}>
            {row}
          </p>
        ))}
      </aside>
      <div className="min-w-0 flex-1">
        <div className="flex h-10 bg-[#181818] font-sans text-[14px]">
          {tabs.map((tab) => (
            <span
              key={tab}
              className={`flex items-center border-r border-[#2b2b2b] px-5 ${tab === open ? "border-t-2 border-t-[#0078d4] bg-[#1e1e1e] text-white" : "text-[#9d9d9d]"}`}
            >
              {tab}
            </span>
          ))}
        </div>
        <div className="py-3">
          {lines.map((line, i) => (
            <p key={i} className={`flex whitespace-pre ${added.includes(i) ? "bg-[#23351f]" : i === cursor ? "bg-[#2a2d2e]" : ""}`}>
              <span className={`w-1 shrink-0 ${added.includes(i) ? "bg-[#487e02]" : ""}`} />
              <span className="w-12 shrink-0 pr-4 text-right text-[#6e7681]">{i + 1}</span>
              {line.map((run, k) => (
                <span key={k} className={run.tone ? TONES[run.tone] : ""}>
                  {run.text}
                </span>
              ))}
              {i === cursor && <Caret t={t} className="ml-px h-[22px] w-0.5 -translate-y-px bg-[#aeafad]" />}
            </p>
          ))}
        </div>
      </div>
    </div>
  </MacWindow>
);

/** A keystroke badge: the keys, and when they show in scene time. */
type Keystroke = { keys: string[]; t: number };

/** One scene of the setup video. Every time in it is seconds from the scene's start. */
export type Scene = {
  /** The chapter title shown while the scene plays. */
  title: string;
  /** How long the scene runs. */
  length: number;
  /** Camera stops. */
  shots: Shot[];
  /** Pointer stops. */
  pointer: Shot[];
  /** Pointer presses. */
  clicks: number[];
  /** Keystroke badges. */
  keys: Keystroke[];
  /** The window the scene shows at scene time t. */
  view: (t: number) => ReactNode;
};
