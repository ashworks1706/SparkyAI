/**
 * The light the glass sits on: a faint grid, two drifting colour fields, and a wash that keeps
 * text over them readable. Fixed, so every section shares one background rather than stacking its
 * own blurred layers.
 */
const Ambient = () => (
  <div
    aria-hidden
    className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
  >
    <div className="absolute inset-0 bg-[#fbf9f8]" />
    <div className="absolute inset-0 bg-grid-faint bg-grid-16 [mask-image:radial-gradient(ellipse_at_center,black,transparent_78%)]" />
    <div className="absolute -left-[14%] -top-[16%] h-[36rem] w-[36rem] rounded-full bg-sparky-maroon/[0.16] blur-[110px] motion-safe:animate-drift-slow" />
    <div
      className="absolute -right-[12%] top-[6%] h-[30rem] w-[30rem] rounded-full bg-sparky-maroon/[0.09] blur-[120px] motion-safe:animate-drift-slow"
      style={{ animationDelay: "-7s" }}
    />
    <div
      className="absolute bottom-[-18%] left-[28%] h-[34rem] w-[34rem] rounded-full bg-sparky-maroon/[0.10] blur-[130px] motion-safe:animate-drift-slow"
      style={{ animationDelay: "-14s" }}
    />
    <div className="absolute inset-0 bg-gradient-to-b from-white/55 via-transparent to-white/75" />
  </div>
);

export default Ambient;
