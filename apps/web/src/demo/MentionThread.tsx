import type { ReactNode } from "react";
import { threadName } from "@/components/discord/format";
import Message from "@/components/discord/Message";
import { SPARKY, type Author, type LinkButton } from "@/components/discord/people";
import Window, { ThreadChip, type Server } from "@/components/discord/Window";

/** Width of the thread panel once open. */
const THREAD_W = 600;

/** Sparky's message in the thread: the card, its buttons, and the button under the pointer. */
type Reply = { text: string; buttons: LinkButton[]; hovered?: number };

/** A mention in the server channel, answered in a thread that opens beside it. */
const MentionThread = ({
  server,
  topic,
  asker,
  time,
  question,
  earlier,
  composer,
  asked,
  chip,
  opening,
  reply,
}: {
  /** The server the channel is in, the landing page server when unset. */
  server?: Server;
  /** The channel topic in the header. */
  topic: string;
  /** The student who mentions Sparky. */
  asker: Author;
  /** The timestamp on both messages. */
  time: string;
  /** The question after the mention. */
  question: string;
  /** Messages above the mention. */
  earlier?: ReactNode;
  /** The message box under the channel. */
  composer: ReactNode;
  /** Whether the mention is posted. */
  asked: boolean;
  /** Whether the thread chip shows under the mention. */
  chip: boolean;
  /** How far the thread panel is open, 0 to 1. */
  opening: number;
  /** Sparky's message, or null before it is posted. */
  reply: Reply | null;
}) => {
  const name = threadName(question);
  const mention = `@Sparky ${question}`;
  return (
    <Window
      server={server}
      channel="ask-sparky"
      topic={topic}
      me={asker}
      messages={
        <>
          {earlier}
          {asked && (
            <Message author={asker} time={time} text={mention}>
              {chip && <ThreadChip name={name} count={reply ? "1 Message" : "See Thread"} />}
            </Message>
          )}
        </>
      }
      composer={composer}
      thread={
        opening > 0
          ? {
              name,
              width: THREAD_W * opening,
              body: (
                <div style={{ width: THREAD_W }}>
                  <Message author={asker} time={time} text={mention} />
                  {reply && <Message author={SPARKY} time={time} text={reply.text} buttons={reply.buttons} hovered={reply.hovered} />}
                </div>
              ),
            }
          : undefined
      }
    />
  );
};

export default MentionThread;
