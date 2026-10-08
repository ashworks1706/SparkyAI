import Thread from "@/components/discord/Thread";
import { EXAMPLES } from "@/components/discord/examples";
import Heading from "./Heading";

/** The sources Sparky reaches, each shown as the thread a question opens on Discord. */
const Sources = () => (
  <section id="sources" aria-label="What students ask" className="relative z-10 bg-paper py-20">
    <div className="mx-auto max-w-6xl px-5 sm:px-8">
      <Heading title="Everything you would otherwise go digging for." />
      <p className="mt-4 text-sm text-ink-soft">Example threads, drawn in the format the bot posts.</p>
      <ul className="mt-12 grid items-start gap-x-8 gap-y-12 lg:grid-cols-2">
        {EXAMPLES.map((example) => (
          <li key={example.title} className="min-w-0">
            <h3 className="mb-3 font-serif text-xl font-bold">{example.title}</h3>
            <Thread example={example} />
          </li>
        ))}
      </ul>
    </div>
  </section>
);

export default Sources;
