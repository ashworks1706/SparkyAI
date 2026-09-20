import { useEffect, useState } from "react";
import GithubMark from "@/components/brand/GithubMark";
import { Link } from "react-router-dom";
import { BrandMark } from "@/components/brand/BrandLogo";
import { REPO, ROADMAP } from "./content";

const LINKS = [
  { label: "In action", href: "#in-action" },
  { label: "Use cases", href: "#use-cases" },
  { label: "How it works", href: "#how-it-works" },
];

const Nav = () => {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="fixed inset-x-0 top-0 z-30 px-4 pt-4 sm:px-8 sm:pt-5">
      <nav
        aria-label="Primary navigation"
        className={`mx-auto flex h-14 max-w-5xl items-center justify-between rounded-full border px-4 backdrop-blur-xl transition-all duration-300 sm:px-5 ${
          scrolled
            ? "border-white/70 bg-white/70 shadow-glass"
            : "border-transparent bg-white/30 shadow-none"
        }`}
      >
        <Link
          to="/"
          className="flex items-center gap-2.5"
          aria-label="SparkyAI home"
        >
          <BrandMark className="h-8 w-auto" decorative />
          <span className="text-base font-semibold tracking-[-0.02em] text-stone-900">
            SparkyAI
          </span>
        </Link>

        <div className="hidden items-center gap-7 text-sm text-stone-600 md:flex">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="transition-colors hover:text-sparky-maroon"
            >
              {link.label}
            </a>
          ))}
          <a
            href={ROADMAP}
            target="_blank"
            rel="noreferrer"
            className="transition-colors hover:text-sparky-maroon"
          >
            Roadmap
          </a>
        </div>

        <div className="flex items-center gap-3">
          <Link
            to="/old"
            className="hidden text-sm text-stone-500 transition-colors hover:text-sparky-maroon sm:block"
          >
            Legacy
          </Link>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="grid h-9 w-9 place-items-center rounded-full border border-white/70 bg-white/60 text-stone-700 shadow-glass transition-colors hover:text-sparky-maroon"
          >
            <GithubMark className="h-4 w-4" />
          </a>
        </div>
      </nav>
    </header>
  );
};

export default Nav;
