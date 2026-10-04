import Cloud from "@/components/dragon/Cloud";

/** The gap between two sections, where the dragon crosses the page among clouds. */
const Crossing = ({ flip = false }: { flip?: boolean }) => (
  <div data-dragon="cross" aria-hidden className="relative h-64 sm:h-96">
    <Cloud
      className={`absolute top-6 w-32 opacity-90 motion-safe:animate-[drift_14s_ease-in-out_infinite] sm:w-52 ${flip ? "right-[12%]" : "left-[12%]"}`}
    />
    <Cloud
      className={`absolute bottom-6 w-24 opacity-70 motion-safe:animate-[drift_18s_ease-in-out_infinite_reverse] sm:w-40 ${flip ? "left-[18%]" : "right-[18%]"}`}
    />
  </div>
);

export default Crossing;
