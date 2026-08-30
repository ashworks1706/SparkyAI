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
    className="border-t border-stone-200 bg-white px-7 py-16 text-[#1c1917] sm:px-10 sm:py-20 md:px-16 lg:px-24"
  >
    <div className="mx-auto max-w-3xl">
      <p className="mb-8 text-[0.65rem] font-medium uppercase tracking-[0.24em] text-stone-500 sm:text-xs">
        README
      </p>
      <div className="prose prose-stone max-w-none prose-headings:font-editorial prose-headings:font-medium prose-headings:tracking-[-0.03em] prose-h1:text-3xl sm:prose-h1:text-4xl prose-h2:mt-12 prose-h2:text-2xl prose-a:text-[#6f1635] prose-a:no-underline hover:prose-a:underline prose-code:rounded prose-code:bg-stone-100 prose-code:px-1 prose-code:py-0.5 prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-pre:bg-stone-950 prose-img:rounded-lg prose-img:border prose-img:border-stone-200">
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
