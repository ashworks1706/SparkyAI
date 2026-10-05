import { finishedCard, threadName } from "./format";
import Message from "./Message";
import { SPARKY } from "./people";
import { button, stepsOf, type Example } from "./scenes";
import { ThreadIcon } from "./Window";

/** A thread as Discord shows it once Sparky has answered: the question, then the reply. */
const Thread = ({ example }: { example: Example }) => (
  <div className="overflow-hidden rounded-xl bg-[#313338] font-sans text-[0.9375rem] leading-[1.375rem] text-[#dbdee1] shadow-[0_30px_60px_-30px_rgba(20,18,16,0.55)] ring-1 ring-black/30">
    <header className="flex h-11 items-center gap-2 border-b border-[#1f2023] px-4">
      <ThreadIcon className="h-5 w-5 shrink-0 text-[#80848e]" />
      <span className="truncate font-semibold text-[#f2f3f5]">{threadName(example.question)}</span>
    </header>
    <div className="space-y-3 py-4">
      <Message author={example.asker} time="Today at 4:02 PM" text={`@Sparky ${example.question}`} />
      <Message
        author={SPARKY}
        time="Today at 4:02 PM"
        text={finishedCard(stepsOf(example), example.answer)}
        buttons={example.calls.map((c) => button(c.source))}
      />
    </div>
  </div>
);

export default Thread;
