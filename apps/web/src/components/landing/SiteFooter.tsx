import GithubMark from "@/components/brand/GithubMark";
import { Link } from "react-router-dom";
import { BrandMark } from "@/components/brand/BrandLogo";
import { REPO } from "./content";

/** One line at the bottom: who made it, what it is not, and the way to the source. */
const SiteFooter = () => (
  <footer className="border-t border-white/60 bg-white/40 px-5 py-5 backdrop-blur-xl sm:px-8 lg:px-10">
    <div className="mx-auto flex max-w-6xl flex-col items-center gap-3 text-xs text-stone-500 sm:flex-row sm:justify-between">
      <div className="flex items-center gap-2">
        <BrandMark className="h-5 w-auto" decorative />
        <span className="font-semibold text-stone-700">SparkyAI</span>
        <span className="hidden sm:inline">
          &middot; not affiliated with Arizona State University
        </span>
      </div>

      <div className="flex items-center gap-5">
        <Link to="/old" className="transition-colors hover:text-sparky-maroon">
          Legacy
        </Link>
        <a
          href={REPO}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 transition-colors hover:text-sparky-maroon"
        >
          <GithubMark className="h-3.5 w-3.5" />
          GitHub
        </a>
      </div>
    </div>
  </footer>
);

export default SiteFooter;
