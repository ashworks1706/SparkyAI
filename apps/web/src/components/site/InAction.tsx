import DemoVideo from "./DemoVideo";
import Heading from "./Heading";

/** A recorded turn: a mention opens a thread, the steps tick by, and the answer lands with its sources. */
const InAction = () => (
  <section id="in-action" aria-label="Sparky in action" className="relative z-10 bg-paper py-20">
    <div className="mx-auto max-w-6xl px-5 sm:px-8">
      <Heading title="It shows its working." />
      <div className="mt-12">
        <DemoVideo name="sparky-demo" label="Sparky answering a question in a Discord thread" />
      </div>
    </div>
  </section>
);

export default InAction;
