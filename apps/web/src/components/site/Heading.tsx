import { motion } from "motion/react";

type HeadingProps = {
  title: string;
  align?: "left" | "center";
};

/** A section heading whose words rise in one after another when it scrolls into view. */
const Heading = ({ title, align = "left" }: HeadingProps) => {
  const center = align === "center";
  return (
    <div className={center ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}>
      <h2 className="text-balance font-serif text-4xl font-bold leading-[1.08] tracking-[-0.02em] sm:text-5xl lg:text-[3.4rem]">
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
    </div>
  );
};

export default Heading;
