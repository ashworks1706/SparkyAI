import { Fragment, type ReactNode } from "react";

/** Inline spans: bold, code, and a mention of the bot. */
const inline = (text: string, key: string): ReactNode[] =>
  text
    .split(/(\*\*[^*]+\*\*|`[^`]+`|@Sparky)/g)
    .filter(Boolean)
    .map((part, i) => {
      const k = `${key}-${i}`;
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={k} className="font-semibold text-[#f2f3f5]">
            {inline(part.slice(2, -2), k)}
          </strong>
        );
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={k} className="rounded-[4px] bg-[#2b2d31] px-[0.2em] py-[0.1em] font-mono text-[0.85em]">
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part === "@Sparky") {
        return (
          <span key={k} className="rounded-[3px] bg-[#5865f2]/30 px-0.5 font-medium text-[#c9cdfb]">
            @Sparky
          </span>
        );
      }
      return <Fragment key={k}>{part}</Fragment>;
    });

/** Message content in the subset of Discord markdown the bot writes: subtext, lists, bold and code. */
const Markdown = ({ text }: { text: string }) => (
  <div className="[overflow-wrap:anywhere]">
    {text.split("\n").map((line, i) => {
      const key = `l${i}`;
      if (line.startsWith("-# ")) {
        return (
          <div key={key} className="text-[0.8125rem] leading-[1.15rem] text-[#949ba4]">
            {inline(line.slice(3), key)}
          </div>
        );
      }
      if (line.startsWith("- ")) {
        return (
          <div key={key} className="flex gap-2 pl-1">
            <span aria-hidden>&bull;</span>
            <span>{inline(line.slice(2), key)}</span>
          </div>
        );
      }
      if (line === "") return <div key={key} className="h-[1.375em]" />;
      return <div key={key}>{inline(line, key)}</div>;
    })}
  </div>
);

export default Markdown;
