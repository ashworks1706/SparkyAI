import Footer from "@/components/site/Footer";
import Hero from "@/components/site/Hero";
import HowItWorks from "@/components/site/HowItWorks";
import InAction from "@/components/site/InAction";
import Nav from "@/components/site/Nav";
import OpenSource from "@/components/site/OpenSource";
import Sources from "@/components/site/Sources";

/** The landing page. */
const Home = () => (
  <div className="min-h-screen">
    <Nav />
    <main>
      <Hero />
      <InAction />
      <Sources />
      <HowItWorks />
      <OpenSource />
    </main>
    <Footer />
  </div>
);

export default Home;
