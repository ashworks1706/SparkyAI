import { Link } from "react-router-dom";
import Hanko from "@/components/site/Hanko";

/** Any path the site does not serve. */
const NotFound = () => (
  <main className="seigaiha grid min-h-screen place-items-center px-5">
    <div className="rounded-xl border border-ink/10 bg-paper px-10 py-12 text-center">
      <Hanko char="迷" className="mx-auto h-14 text-2xl" />
      <h1 className="mt-6 font-serif text-3xl font-semibold">Page not found</h1>
      <p className="mt-3 text-sm text-ink-soft">This path leads nowhere. The dragon is back home.</p>
      <Link
        to="/"
        className="mt-8 inline-flex h-11 items-center rounded-full bg-ink px-6 text-sm font-medium text-paper transition-colors hover:bg-shu"
      >
        Back to home
      </Link>
    </div>
  </main>
);

export default NotFound;
