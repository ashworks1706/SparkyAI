import { reel } from "../Reel";
import { browse } from "./scenes/browse";
import { clone } from "./scenes/clone";
import { sparkyConsole } from "./scenes/console";
import { discord } from "./scenes/discord";
import { env } from "./scenes/env";
import { pages } from "./scenes/pages";
import { source } from "./scenes/source";
import { toml } from "./scenes/toml";

/** The setup video: from clone to the bot answering on the campus server. */
export const SETUP = reel([clone, browse, pages, source, toml, env, sparkyConsole, discord]);
