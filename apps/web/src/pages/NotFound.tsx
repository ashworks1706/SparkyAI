import { Link } from "react-router-dom";
import Lantern from "@/components/dragon/Lantern";

/** Any path the site does not serve. */
const NotFound = () => (
  <main className="seigaiha grid min-h-screen place-items-center px-5">
    <div className="flex flex-col items-center rounded-md border-2 border-ink bg-paper px-10 pb-12 text-center">
      <Lantern cord={30} size={52} />
      <h1 className="mt-6 font-serif text-3xl font-bold">Page not found</h1>
      <p className="mt-3 text-sm text-ink-soft">The dragon went another way.</p>
      <Link
        to="/"
        className="mt-8 inline-flex h-11 items-center rounded-full bg-shu px-6 text-sm font-semibold text-paper transition-transform hover:-translate-y-0.5"
      >
        Back to home
      </Link>
    </div>
  </main>
);

export default NotFound;
