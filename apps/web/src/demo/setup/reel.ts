import { reel } from "../Reel";
import { clone } from "./scenes/clone";
import { sparkyConsole } from "./scenes/console";
import { discord } from "./scenes/discord";
import { env } from "./scenes/env";
import { source } from "./scenes/source";
import { toml } from "./scenes/toml";

/** The setup video: from clone to the bot answering on the campus server. */
export const SETUP = reel([clone, source, toml, env, sparkyConsole, discord]);
