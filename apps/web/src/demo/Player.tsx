import { useEffect, useState, type ComponentType } from "react";
import { flushSync } from "react-dom";
import { Frame as Ask, LENGTH as ASK } from "./Demo";
import { Frame as Setup, LENGTH as SETUP } from "./Setup";

/** The recordings demo.html can play, by the name in its video query parameter. */
const VIDEOS: Record<string, { Frame: ComponentType<{ t: number }>; length: number }> = {
  ask: { Frame: Ask, length: ASK },
  setup: { Frame: Setup, length: SETUP },
};

/** One frame of the named video, seekable from the recorder through window.seekDemo. */
const Player = () => {
  const query = new URLSearchParams(window.location.search);
  const video = VIDEOS[query.get("video") ?? "ask"] ?? VIDEOS.ask;
  const [t, setT] = useState(() => Number(query.get("t") ?? video.length));
  useEffect(() => {
    (window as unknown as { seekDemo: (s: number) => void }).seekDemo = (s) => flushSync(() => setT(s));
  }, []);
  return <video.Frame t={t} />;
};

export default Player;
