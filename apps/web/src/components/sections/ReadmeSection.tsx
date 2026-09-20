import type { ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import readmeSource from "../../../../../README.md?raw";

const REPO_BLOB = "https://github.com/ashworks1706/SparkyAI/blob/main/";
const REPO_RAW = "https://raw.githubusercontent.com/ashworks1706/SparkyAI/main/";

const isAbsolute = (url: string) => /^(https?:|mailto:|#|\/\/)/.test(url);

const toRepoUrl = (base: string, url?: string) => {
  if (!url) return url;
  return isAbsolute(url) ? url : `${base}${url.replace(/^\.?\//, "")}`;
};

/** The repository README, rendered below the landing hero. */
const ReadmeSection = () => (
  <section
    id="readme"
    aria-label="Project readme"
    className="relative px-5 py-20 sm:px-8 sm:py-28 lg:px-10"
  >
    <div className="mx-auto max-w-3xl">
      <p className="mb-6 text-xs font-semibold uppercase tracking-[0.22em] text-sparky-maroon">
        Readme
      </p>
      <div className="prose prose-stone max-w-none rounded-3xl border border-white/70 bg-white/55 px-6 py-8 shadow-glass backdrop-blur-xl prose-headings:font-semibold prose-headings:tracking-[-0.03em] prose-h1:text-3xl sm:prose-h1:text-4xl prose-h2:mt-12 prose-h2:text-2xl prose-a:text-sparky-maroon prose-a:no-underline hover:prose-a:underline prose-code:rounded prose-code:bg-stone-900/[0.06] prose-code:px-1 prose-code:py-0.5 prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-pre:bg-stone-950 prose-img:rounded-lg prose-img:border prose-img:border-stone-200 sm:px-10 sm:py-12">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={{
            a: ({ href, ...props }: ComponentPropsWithoutRef<"a">) => (
              <a
                {...props}
                href={toRepoUrl(REPO_BLOB, href)}
                target="_blank"
                rel="noreferrer"
              />
            ),
            img: ({ src, ...props }: ComponentPropsWithoutRef<"img">) => (
              <img
                {...props}
                src={typeof src === "string" ? toRepoUrl(REPO_RAW, src) : src}
                loading="lazy"
              />
            ),
          }}
        >
          {readmeSource}
        </ReactMarkdown>
      </div>
    </div>
  </section>
);

export default ReadmeSection;
