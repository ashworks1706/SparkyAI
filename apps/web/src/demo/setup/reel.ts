import { browse, clone, env, pages, source, toml } from "./code";
import { sparkyConsole } from "./console";
import { discord } from "./discord";
import type { Scene } from "./kit";

/** The scenes in order: from clone to the bot answering on the campus server. */
export const SCENES: Scene[] = [clone, browse, pages, source, toml, env, sparkyConsole, discord];

/** When each scene starts, in seconds. */
export const STARTS = SCENES.reduce<number[]>((starts, _, i) => [...starts, i === 0 ? 0 : starts[i - 1] + SCENES[i - 1].length], []);

/** Length of the setup video, in seconds. */
export const LENGTH = Math.round(STARTS[STARTS.length - 1] + SCENES[SCENES.length - 1].length);
