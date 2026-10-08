import type { Scene } from "../../timeline";
import { REST } from "../editor";
import { Terminal, type Line } from "../kit";

/** Clone the repository and bootstrap it. Output is what git and the justfile print. */
const CLONE: Line[] = [
  { at: 0.5, to: 2.0, text: "git clone https://github.com/ashworks1706/SparkyAI && cd SparkyAI" },
  { at: 2.2, text: "Cloning into 'SparkyAI'...", tone: "dim" },
  { at: 2.5, text: "Receiving objects: 100% (9418/9418), 6.12 MiB | 18.4 MiB/s, done.", tone: "dim" },
  { at: 2.8, text: "Resolving deltas: 100% (5873/5873), done.", tone: "dim" },
  { at: 3.1, to: 3.7, text: "just bootstrap" },
  { at: 3.9, text: "created .env, fill in tokens and model URLs" },
  { at: 4.1, text: "hooks installed: .githooks/pre-commit" },
  { at: 4.4, text: " ✔ Container deploy-postgres-1  Healthy", tone: "ok" },
  { at: 4.5, text: " ✔ Container deploy-redis-1     Healthy", tone: "ok" },
  { at: 4.6, text: " ✔ Container deploy-minio-1     Healthy", tone: "ok" },
  { at: 4.8, text: "applied: 0001_init, 0002_query_sources, 0003_summary_turns, …" },
  { at: 5.0, text: "ready. Next a model, because the engine answers nothing without one:" },
  { at: 5.1, text: "  GPU:    just model       llama-server on CUDA", tone: "dim" },
  { at: 5.2, text: "  no GPU: just model-cpu   the same models on the processor, slowly", tone: "dim" },
  { at: 5.3, text: "  hosted: set SPARKY_MODEL__BASE_URL and SPARKY_MODEL__API_KEY in .env", tone: "dim" },
];

/** Scene: clone the repository and bootstrap it in a terminal. */
export const clone: Scene = {
  title: "Clone and bootstrap",
  length: 5.8,
  shots: [
    { t: 0, x: 800, y: 500, s: 1 },
    { t: 0.4, x: 800, y: 500, s: 1 },
    { t: 1.1, x: 640, y: 300, s: 1.35 },
    { t: 3.4, x: 640, y: 380, s: 1.35 },
    { t: 5.4, x: 700, y: 500, s: 1.2 },
  ],
  pointer: [REST],
  clicks: [],
  keys: [],
  view: (t) => <Terminal lines={CLONE} t={t} dir={(i) => (i === 0 ? "~" : "~/SparkyAI")} />,
};
