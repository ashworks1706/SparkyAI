import type { ReactNode } from "react";
import Markdown from "./Markdown";
import type { Author, LinkButton } from "./people";

/** Discord's open in new tab glyph on link buttons. */
const External = () => (
  <svg aria-hidden viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
    <path d="M15 2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v6a1 1 0 1 1-2 0V4.41l-4.3 4.3a1 1 0 1 1-1.4-1.42L19.58 3H16a1 1 0 0 1-1-1Z" />
    <path d="M5 2a3 3 0 0 0-3 3v14a3 3 0 0 0 3 3h14a3 3 0 0 0 3-3v-6a1 1 0 1 0-2 0v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h6a1 1 0 1 0 0-2H5Z" />
  </svg>
);

/** A round avatar: the logo for the bot, an initial on a colour for a person. */
export const Avatar = ({ author, size = 40 }: { author: Author; size?: number }) =>
  author.bot ? (
    <span className="flex shrink-0 items-center justify-center rounded-full bg-[#1e1f22]" style={{ width: size, height: size }}>
      <img src="/brand/sparkyai-logo.png" alt="" className="h-[70%] w-auto" />
    </span>
  ) : (
    <span
      className="flex shrink-0 items-center justify-center rounded-full font-semibold text-white"
      style={{ width: size, height: size, background: author.color ?? "#5865f2", fontSize: size * 0.42 }}
    >
      {author.initial ?? author.name[0].toUpperCase()}
    </span>
  );

/** One message as Discord lays it out: avatar, name, time, content, then buttons. */
const Message = ({
  author,
  time,
  text,
  buttons = [],
  hovered,
  children,
}: {
  author: Author;
  time: string;
  text?: string;
  buttons?: LinkButton[];
  hovered?: number;
  children?: ReactNode;
}) => (
  <div className="flex gap-4 px-4 py-1">
    <Avatar author={author} />
    <div className="min-w-0 flex-1">
      <p className="flex items-center gap-1.5 leading-[1.375rem]">
        <span className="font-medium" style={{ color: author.bot ? "#f2f3f5" : (author.color ?? "#f2f3f5") }}>
          {author.name}
        </span>
        {author.bot && (
          <span className="rounded-[3px] bg-[#5865f2] px-1 text-[0.625rem] font-semibold leading-[0.95rem] text-white">
            APP
          </span>
        )}
        <span className="ml-1 text-xs text-[#949ba4]">{time}</span>
      </p>
      {text !== undefined && <Markdown text={text} />}
      {children}
      {buttons.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {buttons.map((b, i) => (
            <a
              key={b.label}
              href={b.href}
              target="_blank"
              rel="noreferrer"
              className={`inline-flex h-8 items-center gap-2 rounded-[8px] px-4 text-sm font-medium text-white ${i === hovered ? "bg-[#6d6f78]" : "bg-[#4e5058]"}`}
            >
              {b.label}
              <External />
            </a>
          ))}
        </div>
      )}
    </div>
  </div>
);

export default Message;
