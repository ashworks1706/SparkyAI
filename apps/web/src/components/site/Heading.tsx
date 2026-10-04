import type { ReactNode } from "react";
import { motion } from "motion/react";

type HeadingProps = {
  /** Small label above the title. */
  label: string;
  /** Ornamental numeral beside the label. */
  numeral: string;
  title: string;
  lead?: ReactNode;
  align?: "left" | "center";
};

/** A section heading whose words rise in one after another when it scrolls into view. */
const Heading = ({ label, numeral, title, lead, align = "left" }: HeadingProps) => {
  const center = align === "center";
  return (
    <div className={center ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}>
      <p
        className={`flex items-center gap-3 text-xs font-semibold uppercase tracking-[0.3em] text-shu ${center ? "justify-center" : ""}`}
      >
        <span className="grid h-8 w-8 place-items-center rounded-full bg-shu font-serif text-sm tracking-normal text-paper ring-2 ring-kin ring-offset-2 ring-offset-paper">
          {numeral}
        </span>
        {label}
      </p>
      <h2 className="mt-6 text-balance font-serif text-4xl font-bold leading-[1.08] tracking-[-0.02em] sm:text-5xl lg:text-[3.4rem]">
        {title.split(" ").map((word, i) => (
          <span key={i} className="inline-block overflow-hidden pb-[0.14em] align-bottom">
            <motion.span
              className="inline-block"
              initial={{ y: "105%" }}
              whileInView={{ y: 0 }}
              viewport={{ once: true, margin: "-10% 0px" }}
              transition={{ duration: 0.7, delay: i * 0.06, ease: [0.22, 1, 0.36, 1] }}
            >
              {word}&nbsp;
            </motion.span>
          </span>
        ))}
      </h2>
      {lead && (
        <motion.p
          className={`mt-6 text-pretty text-lg leading-8 text-ink-soft ${center ? "mx-auto max-w-2xl" : "max-w-2xl"}`}
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
          {lead}
        </motion.p>
      )}
    </div>
  );
};

export default Heading;
