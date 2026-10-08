import type { Scene } from "../../timeline";
import { CAMPUS } from "../campus";
import { MacWindow } from "../kit";

/** The registrar page the developer takes a URL from. */
const browserAt = (selected: boolean) => (
  <MacWindow
    dark={false}
    title={
      <span className="mt-0.5 flex h-8 w-[560px] items-center gap-2 rounded-lg bg-white px-4 text-[15px] text-[#3c3c43] ring-1 ring-black/10">
        <svg aria-hidden viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
          <path d="M7 10V7a5 5 0 0 1 10 0v3h1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V11a1 1 0 0 1 1-1h1Zm2 0h6V7a3 3 0 0 0-6 0v3Z" />
        </svg>
        <span className={selected ? "rounded-sm bg-[#b3d7ff]" : ""}>{CAMPUS.page.replace("https://", "")}</span>
      </span>
    }
  >
    <div className="h-full bg-white font-sans text-[#1b1f2a]">
      <div className="flex h-16 items-center gap-4 bg-[#1d3557] px-10 text-white">
        <span className="font-serif text-xl font-bold">{CAMPUS.name}</span>
        <span className="text-white/60">|</span>
        <span className="text-white/80">Office of the Registrar</span>
      </div>
      <div className="mx-auto max-w-4xl px-10 py-10">
        <h1 className="font-serif text-4xl font-bold">Academic Calendar</h1>
        <p className="mt-2 text-[#5b6170]">Spring 2027 dates and deadlines.</p>
        <table className="mt-8 w-full text-left text-lg">
          <tbody className="divide-y divide-[#e3e6ec]">
            {[
              ["Classes begin", "January 11"],
              ["Last day to add or drop", "January 19"],
              ["Spring break", "March 8 to 14"],
              ["Last day of classes", "April 30"],
              ["Final exams", "May 3 to 8"],
            ].map(([what, when]) => (
              <tr key={what}>
                <td className="py-3.5">{what}</td>
                <td className="py-3.5 font-semibold">{when}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </MacWindow>
);

export const browse: Scene = {
  title: "Pick a page from your campus site",
  length: 3.8,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.3, x: 800, y: 500, s: 1 },
    { t: 1.0, x: 520, y: 150, s: 1.9 },
    { t: 2.9, x: 520, y: 150, s: 1.9 },
    { t: 3.6, x: 800, y: 500, s: 1 },
  ],
  pointer: [
    { t: 0, x: 1180, y: 700, s: 1 },
    { t: 0.5, x: 1000, y: 560, s: 1 },
    { t: 1.4, x: 560, y: 152, s: 1 },
  ],
  clicks: [1.5],
  keys: [{ keys: ["⌘", "C"], t: 1.9 }],
  view: (t) => browserAt(t >= 1.55),
};
