import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { finishedCard, runningCard, thinkingStep, thoughtStep, threadName } from "@/components/discord/format";
import Message, { Avatar } from "@/components/discord/Message";
import { SPARKY, type Author } from "@/components/discord/people";
import { DEMO, button, done, started } from "@/components/discord/scenes";
import Window, { Composer, ThreadIcon } from "@/components/discord/Window";

/** Stage size the video is recorded at. */
const STAGE = { w: 1600, h: 1000 };

/** Length of the demo, in seconds. */
export const LENGTH = 18;

/** Where the window sits on the stage. */
const FRAME = { x: 80, y: 70, w: 1440, h: 860 };

/** Width of the thread panel once open. */
const THREAD_W = 460;

/** When each beat of the demo happens, in seconds. Edits of the bot card are 1.5 s apart, as bot.edit_every_ms. */
const AT = {
  mention: 1.6,
  pick: 2.15,
  type: 2.35,
  typed: 4.6,
  send: 4.9,
  thread: 5.2,
  open: 5.8,
  edits: [6.2, 7.7, 9.2, 10.7, 12.2],
  finished: 13.7,
  hover: 14.7,
};

/** A camera stop: what point of the stage is centred and how far in. */
type Shot = { t: number; x: number; y: number; s: number };

/** Camera stops; the camera eases between consecutive ones. */
const SHOTS: Shot[] = [
  { t: 0, x: 800, y: 500, s: 1 },
  { t: 1.1, x: 800, y: 500, s: 1 },
  { t: 1.9, x: 700, y: 720, s: 1.55 },
  { t: 4.9, x: 700, y: 720, s: 1.55 },
  { t: 5.6, x: 800, y: 500, s: 1 },
  { t: 6.4, x: 800, y: 500, s: 1 },
  { t: 7.2, x: 1290, y: 620, s: 1.5 },
  { t: 13.7, x: 1290, y: 620, s: 1.5 },
  { t: 14.5, x: 1290, y: 680, s: 1.6 },
  { t: 16.2, x: 1290, y: 680, s: 1.6 },
  { t: 17.2, x: 800, y: 500, s: 1 },
];

/** Cursor stops on the stage. */
const POINTER: Shot[] = [
  { t: 0, x: 1040, y: 560, s: 1 },
  { t: 1.4, x: 640, y: 884, s: 1 },
  { t: 1.9, x: 640, y: 884, s: 1 },
  { t: 2.1, x: 520, y: 836, s: 1 },
  { t: 2.4, x: 600, y: 900, s: 1 },
  { t: 5.2, x: 600, y: 900, s: 1 },
  { t: 5.75, x: 690, y: 826, s: 1 },
  { t: 6.4, x: 690, y: 826, s: 1 },
  { t: 7.4, x: 1380, y: 520, s: 1 },
  { t: 13.9, x: 1380, y: 520, s: 1 },
  { t: 14.7, x: 1222, y: 770, s: 1 },
];

/** Clicks: when the pointer presses. */
const CLICKS = [1.5, 2.12, AT.open];

const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);

/** The interpolated stop at time t. */
const at = (stops: Shot[], t: number): Shot => {
  const next = stops.findIndex((s) => s.t > t);
  if (next === -1) return stops[stops.length - 1];
  if (next === 0) return stops[0];
  const a = stops[next - 1];
  const b = stops[next];
  const u = ease((t - a.t) / (b.t - a.t));
  return { t, x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, s: a.s + (b.s - a.s) * u };
};

/** 0 before from, 1 after from + over, eased between. */
const progress = (t: number, from: number, over: number) => ease(Math.min(1, Math.max(0, (t - from) / over)));

const TIME = "Today at 4:02 PM";

/** Other people in the channel before the demo starts. */
const BEFORE: { author: Author; text: string }[] = [
  { author: { name: "devon", color: "#23a55a" }, text: "anyone else lose wifi in the library basement" },
  { author: { name: "sam", color: "#00a8fc" }, text: "yeah it's been spotty all week" },
];

/** A question someone asked Sparky earlier, with the thread it opened. */
const EARLIER = { author: { name: "ren", color: "#eb459e" }, question: "when is the last day to drop a class this fall?" };

/** What Sparky's card says at time t, or null before it is posted. */
const card = (t: number) => {
  const [first, second, third, fourth, fifth] = AT.edits;
  const [hours, rooms] = DEMO.calls;
  const thought = thoughtStep(DEMO.thought);
  const doneSteps = [thought, done(hours), done(rooms)];
  if (t >= AT.finished) return finishedCard([...doneSteps, thoughtStep(DEMO.wrapUp)], DEMO.answer);
  if (t >= fifth) return runningCard([...doneSteps, thoughtStep(DEMO.wrapUp)], 4, DEMO.answer.split("\n\n")[0]);
  if (t >= fourth) return runningCard([...doneSteps, thinkingStep()], 3);
  if (t >= third) return runningCard(doneSteps, 2);
  if (t >= second) return runningCard([thought, started(hours), started(rooms)], 1);
  if (t >= first) return runningCard([thinkingStep()], 0);
  return null;
};

/** What the composer holds at time t: nothing, the mention being typed, or the question. */
const draft = (t: number) => {
  if (t < AT.mention || t >= AT.send) return null;
  const caret = Math.floor(t * 2.2) % 2 === 0;
  if (t < AT.pick) {
    const typed = "@Sparky".slice(0, 1 + Math.floor((t - AT.mention) / 0.12));
    return { mention: false, text: typed.slice(0, 3), caret };
  }
  const chars = Math.floor(DEMO.question.length * Math.min(1, Math.max(0, (t - AT.type) / (AT.typed - AT.type))));
  return { mention: true, text: DEMO.question.slice(0, chars), caret };
};

/** The pointer Screen Studio style recordings show, scaled with the camera. */
const Pointer = ({ x, y, pressed }: { x: number; y: number; pressed: number }) => (
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

/** The box under a message that opened a thread. */
const ThreadChip = ({ name, count }: { name: string; count: string }) => (
  <div className="mt-1.5 flex max-w-md items-center gap-2 rounded-lg bg-[#2b2d31] px-3 py-2 text-sm">
    <ThreadIcon className="h-4 w-4 shrink-0 text-[#b5bac1]" />
    <span className="truncate font-semibold text-[#f2f3f5]">{name}</span>
    <span className="shrink-0 font-semibold text-[#00a8fc]">{count} &rsaquo;</span>
  </div>
);

/** One frame of the demo at time t. */
export const Frame = ({ t }: { t: number }) => {
  const shot = at(SHOTS, t);
  const pointer = at(POINTER, t);
  const click = CLICKS.map((c) => (t >= c && t < c + 0.5 ? (t - c) / 0.5 : 0)).find((p) => p > 0) ?? 0;
  const composing = draft(t);
  const sent = t >= AT.send;
  const opening = progress(t, AT.open, 0.4);
  const chip = progress(t, AT.thread, 0.3);
  const reply = card(t);
  const name = threadName(DEMO.question);
  const hover = t >= AT.hover;

  const sparkyReply = reply && (
    <Message
      author={SPARKY}
      time={TIME}
      text={reply}
      buttons={t >= AT.finished ? [button(DEMO.calls[0].source), button(DEMO.calls[1].source)] : []}
      hovered={hover ? 0 : undefined}
    />
  );

  const messages = (
    <>
      <Message author={EARLIER.author} time="Today at 3:41 PM" text={`@Sparky ${EARLIER.question}`}>
        <ThreadChip name={threadName(EARLIER.question)} count="3 Messages" />
      </Message>
      {BEFORE.map((m) => (
        <Message key={m.text} author={m.author} time="Today at 3:58 PM" text={m.text} />
      ))}
      {sent && (
        <Message author={DEMO.asker} time={TIME} text={`@Sparky ${DEMO.question}`}>
          {t >= AT.thread && (
            <div style={{ opacity: chip }}>
              <ThreadChip name={name} count={reply ? "1 Message" : "See Thread"} />
            </div>
          )}
        </Message>
      )}
    </>
  );

  const composer = (
    <div className="relative">
      {composing && !composing.mention && composing.text.length >= 2 && (
        <div className="absolute inset-x-4 bottom-full mb-2 rounded-lg bg-[#2b2d31] p-2 shadow-xl ring-1 ring-black/30">
          <p className="px-2 pb-1 text-xs font-semibold uppercase text-[#b5bac1]">Members</p>
          <div className="flex items-center gap-2 rounded bg-[#404249] px-2 py-1.5">
            <Avatar author={SPARKY} size={24} />
            <span className="font-medium text-[#f2f3f5]">Sparky</span>
            <span className="rounded-[3px] bg-[#5865f2] px-1 text-[0.625rem] font-semibold leading-[0.95rem] text-white">APP</span>
          </div>
        </div>
      )}
      <Composer placeholder="Message #ask-sparky">
        {composing && (
          <span>
            {composing.mention ? (
              <span className="rounded-[3px] bg-[#5865f2]/30 px-0.5 font-medium text-[#c9cdfb]">@Sparky</span>
            ) : (
              composing.text
            )}
            {composing.mention && ` ${composing.text}`}
            <span className={`ml-px inline-block h-5 w-0.5 translate-y-1 bg-[#dbdee1] ${composing.caret ? "" : "opacity-0"}`} />
          </span>
        )}
      </Composer>
    </div>
  );

  const thread =
    opening > 0
      ? {
          name,
          width: THREAD_W * opening,
          body: (
            <div style={{ width: THREAD_W }}>
              <Message author={DEMO.asker} time={TIME} text={`@Sparky ${DEMO.question}`} />
              {sparkyReply}
            </div>
          ),
        }
      : undefined;

  return (
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
        <div className="absolute" style={{ left: FRAME.x, top: FRAME.y, width: FRAME.w, height: FRAME.h }}>
          <Window
            channel="ask-sparky"
            topic="Ask Sparky about classes, hours, shuttles and more. Mention it to start a thread."
            me={DEMO.asker}
            messages={messages}
            composer={composer}
            thread={thread}
          />
        </div>
        <Pointer x={pointer.x} y={pointer.y} pressed={click} />
      </div>
    </div>
  );
};

/** The demo page: one frame, seekable from the recorder through window.seekDemo. */
const Demo = () => {
  const [t, setT] = useState(() => Number(new URLSearchParams(window.location.search).get("t") ?? LENGTH));
  useEffect(() => {
    (window as unknown as { seekDemo: (s: number) => void }).seekDemo = (s) => flushSync(() => setT(s));
  }, []);
  return <Frame t={t} />;
};

export default Demo;
