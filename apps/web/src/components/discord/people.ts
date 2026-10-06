/** Who wrote a message, as Discord shows them. */
export type Author = { name: string; bot?: boolean; color?: string; initial?: string };

/** A button under a bot message: a link when it has href, else a press button in its style. */
export type LinkButton = { label: string; href?: string; style?: "danger" | "secondary" };

/** The bot, as its Discord application shows it. */
export const SPARKY: Author = { name: "Sparky", bot: true };
