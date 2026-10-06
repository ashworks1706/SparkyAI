import { finishedCard, runningCard, thinkingStep, thoughtStep, toolDone, toolStarted, type Args } from "@/components/discord/format";

/** One tool call of a turn: the tool, its arguments, and what came back. */
export type Call = { tool: string; args: Args; output: string };

/** A held action: the tool, its arguments as the engine prints them, and the answer once approved. */
export type Held = { tool: string; args: string; done: string };

/** One question and how Sparky answers it. */
export type Turn = {
  question: string;
  thought: string;
  calls: Call[];
  wrap: string;
  answer: string;
  held?: Held;
};

/** Seconds between edits of the card, as bot.edit_every_ms. */
export const EDIT = 1.5;

/** Seconds from the first post of the card to the finished turn. */
export const FINISH = 3 * EDIT;

/** The approval line the bot appends for a held action, as render::card writes it. */
const approval = (held: Held) =>
  `**\`${held.tool}\` needs your approval:** Run \`${held.tool}\` with ${held.args}. This posts or submits externally.`;

/** The steps of the finished turn. */
const steps = (turn: Turn) => [thoughtStep(turn.thought), ...turn.calls.map((c) => toolDone(c.tool, c.args, c.output)), thoughtStep(turn.wrap)];

/** The finished card, with the approval line when an action is held. */
export const finished = (turn: Turn) =>
  finishedCard(steps(turn), turn.held ? `${turn.answer}\n\n${approval(turn.held)}` : turn.answer);

/** The card once the held action was approved: the old steps, the approval, and the result. */
export const approved = (turn: Turn) => finishedCard([...steps(turn), "approved"], turn.held?.done ?? "");

/** The card at t seconds after it was first posted, or null before. */
export const cardAt = (turn: Turn, t: number) => {
  if (t < 0) return null;
  if (t >= FINISH) return finished(turn);
  const thought = thoughtStep(turn.thought);
  if (t >= 2 * EDIT) return runningCard([thought, ...turn.calls.map((c) => toolDone(c.tool, c.args, c.output)), thinkingStep()], 2);
  if (t >= EDIT) return runningCard([thought, ...turn.calls.map((c) => toolStarted(c.tool, c.args))], 1);
  return runningCard([thinkingStep()], 0);
};
