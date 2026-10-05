/** Who wrote a message, as Discord shows them. */
export type Author = { name: string; bot?: boolean; color?: string; initial?: string };

/** A link button under a bot message. */
export type LinkButton = { label: string; href: string };

/** The bot, as its Discord application shows it. */
export const SPARKY: Author = { name: "Sparky", bot: true };
