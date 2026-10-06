import type { ReactNode } from "react";
import { Avatar } from "./Message";
import type { Author } from "./people";

/** The thread glyph Discord puts beside a thread name. */
export const ThreadIcon = ({ className = "h-5 w-5" }: { className?: string }) => (
  <svg aria-hidden viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H11l-4.5 4v-4h0A2.5 2.5 0 0 1 4 13.5v-8Z" />
    <path d="M8.5 8h7M8.5 11.5h4.5" />
  </svg>
);

/** The hash glyph of a text channel. */
const Hash = ({ className = "h-5 w-5" }: { className?: string }) => (
  <svg aria-hidden viewBox="0 0 24 24" className={className} fill="currentColor">
    <path
      fillRule="evenodd"
      d="M10.99 3.16A1 1 0 1 0 9 2.84L8.15 8H4a1 1 0 0 0 0 2h3.82l-.67 4H3a1 1 0 1 0 0 2h3.82l-.8 4.84a1 1 0 0 0 1.97.32L8.85 16h4.97l-.8 4.84a1 1 0 0 0 1.97.32l.86-5.16H20a1 1 0 1 0 0-2h-3.82l.67-4H21a1 1 0 1 0 0-2h-3.82l.8-4.84a1 1 0 1 0-1.97-.32L15.15 8h-4.97l.8-4.84ZM14.15 14l.67-4H9.85l-.67 4h4.97Z"
    />
  </svg>
);

/** Channels in the sidebar; the active one is highlighted. */
const CHANNELS = ["welcome", "general", "ask-sparky", "cs-majors", "housing"];

/** A Discord server: its name and the badge on its icon. */
export type Server = { name: string; badge: string; className: string };

/** The server the landing page demo runs in. */
const STUDENTS: Server = { name: "Sun Devil Students", badge: "ASU", className: "bg-shu text-kin" };

/** Friends listed above the bot in the direct message list. */
const FRIENDS: Author[] = [
  { name: "devon", color: "#23a55a" },
  { name: "sam", color: "#00a8fc" },
];

/** The direct message list, with the open conversation highlighted. */
const DirectMessages = ({ open }: { open: Author }) => (
  <>
    <p className="flex h-12 items-center border-b border-[#1f2023] px-4">
      <span className="w-full rounded bg-[#1e1f22] px-2 py-1 text-sm text-[#949ba4]">Find or start a conversation</span>
    </p>
    <div className="flex-1 px-2 pt-4">
      <p className="mb-1 px-2 text-xs font-semibold uppercase tracking-wide text-[#949ba4]">Direct messages</p>
      {[open, ...FRIENDS].map((a) => (
        <p key={a.name} className={`flex h-11 items-center gap-3 rounded px-2 ${a === open ? "bg-[#404249] text-white" : "text-[#949ba4]"}`}>
          <Avatar author={a} size={32} />
          <span className="font-medium">{a.name}</span>
          {a.bot && <span className="rounded-[3px] bg-[#5865f2] px-1 text-[0.625rem] font-semibold leading-[0.95rem] text-white">APP</span>}
        </p>
      ))}
    </div>
  </>
);

/** The server rail, channel list or direct messages, and user panel of a Discord client. */
const Chrome = ({ channel, me, server, dm }: { channel: string; me: Author; server: Server; dm?: Author }) => (
  <>
    <nav className="flex w-[72px] shrink-0 flex-col items-center gap-2 bg-[#1e1f22] pt-3">
      <span className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-[#5865f2] text-white">
        {dm && <span className="absolute -left-3 h-10 w-1 rounded-r bg-white" />}
        <svg aria-hidden viewBox="0 0 24 24" className="h-7 w-7" fill="currentColor">
          <path d="M19.73 4.87a18.2 18.2 0 0 0-4.6-1.44c-.2.36-.43.85-.59 1.23a16.9 16.9 0 0 0-5.08 0c-.16-.38-.4-.87-.6-1.23-1.6.27-3.14.75-4.6 1.44A19.08 19.08 0 0 0 .96 17.7a18.43 18.43 0 0 0 5.63 2.87c.46-.62.86-1.28 1.2-1.98-.66-.25-1.29-.55-1.88-.9.16-.12.31-.24.46-.36a13.1 13.1 0 0 0 11.26 0l.46.36c-.6.35-1.23.65-1.88.9.35.7.75 1.36 1.2 1.98 2.03-.63 3.94-1.6 5.64-2.87a19 19 0 0 0-3.32-12.83ZM8.3 15.12c-1.1 0-2-1.02-2-2.27 0-1.24.88-2.26 2-2.26s2.02 1.02 2 2.26c0 1.25-.89 2.27-2 2.27Zm7.4 0c-1.1 0-2-1.02-2-2.27 0-1.24.88-2.26 2-2.26s2.02 1.02 2 2.26c0 1.25-.88 2.27-2 2.27Z" />
        </svg>
      </span>
      <span className="h-0.5 w-8 rounded bg-[#35363c]" />
      <span className={`relative flex h-12 w-12 items-center justify-center rounded-2xl font-serif text-sm font-bold ${server.className}`}>
        {!dm && <span className="absolute -left-3 h-10 w-1 rounded-r bg-white" />}
        {server.badge}
      </span>
      <span className="h-12 w-12 rounded-full bg-[#313338]" />
      <span className="h-12 w-12 rounded-full bg-[#313338]" />
    </nav>
    <aside className="flex w-[240px] shrink-0 flex-col bg-[#2b2d31]">
      {dm ? (
        <DirectMessages open={dm} />
      ) : (
        <>
          <p className="flex h-12 items-center border-b border-[#1f2023] px-4 font-semibold text-[#f2f3f5]">{server.name}</p>
          <div className="flex-1 px-2 pt-4">
            <p className="mb-1 px-2 text-xs font-semibold uppercase tracking-wide text-[#949ba4]">Text channels</p>
            {CHANNELS.map((c) => (
              <p
                key={c}
                className={`flex h-8 items-center gap-1.5 rounded px-2 ${c === channel ? "bg-[#404249] text-white" : "text-[#949ba4]"}`}
              >
                <Hash className="h-5 w-5 opacity-70" />
                {c}
              </p>
            ))}
          </div>
        </>
      )}
      <div className="flex h-[52px] items-center gap-2 bg-[#232428] px-2">
        <Avatar author={me} size={32} />
        <div className="leading-tight">
          <p className="text-sm font-semibold text-[#f2f3f5]">{me.name}</p>
          <p className="text-xs text-[#949ba4]">Online</p>
        </div>
      </div>
    </aside>
  </>
);

/** The message box at the bottom of a channel or thread. */
export const Composer = ({ placeholder, children }: { placeholder: string; children?: ReactNode }) => (
  <div className="px-4 pb-6">
    <div className="flex min-h-11 items-center gap-4 rounded-lg bg-[#383a40] px-4 py-2.5">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#b5bac1] text-lg leading-none text-[#383a40]">
        +
      </span>
      <div className="min-w-0 flex-1">{children ?? <span className="block truncate text-[#6d6f78]">{placeholder}</span>}</div>
    </div>
  </div>
);

/** A Discord client on one channel or direct message, with an optional thread open beside it. */
const Window = ({
  channel,
  topic,
  me,
  messages,
  composer,
  thread,
  server = STUDENTS,
  dm,
}: {
  server?: Server;
  dm?: Author;
  channel: string;
  topic: string;
  me: Author;
  messages: ReactNode;
  composer: ReactNode;
  thread?: { name: string; body: ReactNode; width: number };
}) => (
  <div className="flex h-full w-full overflow-hidden rounded-xl bg-[#313338] font-sans text-base text-[#dbdee1] shadow-[0_40px_80px_-30px_rgba(0,0,0,0.6)] ring-1 ring-black/40">
    <Chrome channel={channel} me={me} server={server} dm={dm} />
    <main className="flex min-w-0 flex-1 flex-col">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-[#1f2023] px-4">
        {dm ? <span className="text-2xl leading-none text-[#80848e]">@</span> : <Hash className="h-6 w-6 text-[#80848e]" />}
        <span className="shrink-0 whitespace-nowrap font-semibold text-[#f2f3f5]">{channel}</span>
        <span className="mx-2 h-6 w-px bg-[#3f4147]" />
        <span className="truncate text-sm text-[#949ba4]">{topic}</span>
      </header>
      <div className="flex min-h-0 flex-1 flex-col justify-end overflow-hidden pb-4">{messages}</div>
      {composer}
    </main>
    {thread && (
      <section className="flex shrink-0 flex-col overflow-hidden border-l border-[#1f2023] bg-[#313338]" style={{ width: thread.width }}>
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-[#1f2023] px-4">
          <ThreadIcon className="h-5 w-5 shrink-0 text-[#80848e]" />
          <span className="truncate font-semibold text-[#f2f3f5]">{thread.name}</span>
          <span className="ml-auto text-xl leading-none text-[#b5bac1]">&times;</span>
        </header>
        <div className="flex min-h-0 flex-1 flex-col justify-end overflow-hidden pb-4">{thread.body}</div>
        <Composer placeholder={`Message ${thread.name}`} />
      </section>
    )}
  </div>
);

export default Window;
