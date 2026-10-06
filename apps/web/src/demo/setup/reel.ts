import { reel } from "../Reel";
import { browse, clone, env, pages, source, toml } from "./code";
import { sparkyConsole } from "./console";
import { discord } from "./discord";

/** The setup video: from clone to the bot answering on the campus server. */
export const SETUP = reel([clone, browse, pages, source, toml, env, sparkyConsole, discord]);
