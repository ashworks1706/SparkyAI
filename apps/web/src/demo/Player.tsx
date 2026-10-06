import { useEffect, useState, type ComponentType } from "react";
import { flushSync } from "react-dom";
import { Frame as Ask, LENGTH as ASK } from "./Demo";
import { Frame as Setup } from "./Setup";
import { LENGTH as SETUP } from "./setup/reel";

/** The recordings demo.html can play, by the name in its video query parameter. */
const VIDEOS: Record<string, { Frame: ComponentType<{ t: number }>; length: number }> = {
  ask: { Frame: Ask, length: ASK },
  setup: { Frame: Setup, length: SETUP },
};

/** One frame of the named video, seekable through window.seekDemo, with its length on window.demoLength. */
const Player = () => {
  const query = new URLSearchParams(window.location.search);
  const video = VIDEOS[query.get("video") ?? "ask"] ?? VIDEOS.ask;
  const [t, setT] = useState(() => Number(query.get("t") ?? video.length));
  useEffect(() => {
    const page = window as unknown as { seekDemo: (s: number) => void; demoLength: number };
    page.seekDemo = (s) => flushSync(() => setT(s));
    page.demoLength = video.length;
  }, [video.length]);
  return <video.Frame t={t} />;
};

export default Player;
