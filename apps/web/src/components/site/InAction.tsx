import DemoVideo from "./DemoVideo";
import Heading from "./Heading";

/** A recorded day: deadlines, inbox, campus questions, a booking held for approval, grades and the week ahead. */
const InAction = () => (
  <section id="in-action" aria-label="Sparky in action" className="relative z-10 bg-paper py-20">
    <div className="mx-auto max-w-6xl px-5 sm:px-8">
      <Heading title="It shows its working." />
      <div className="mt-12">
        <DemoVideo name="sparky-demo" label="A student's day with Sparky in Discord, from Canvas deadlines to booking a study room" />
      </div>
    </div>
  </section>
);

export default InAction;
