import Ambient from "@/components/landing/Ambient";
import Hero from "@/components/landing/Hero";
import HowItWorks from "@/components/landing/HowItWorks";
import InAction from "@/components/landing/InAction";
import Nav from "@/components/landing/Nav";
import OpenSource from "@/components/landing/OpenSource";
import SiteFooter from "@/components/landing/SiteFooter";
import UseCases from "@/components/landing/UseCases";
import ReadmeSection from "@/components/sections/ReadmeSection";

const Home = () => (
  <div className="min-h-screen scroll-smooth text-stone-900">
    <Ambient />
    <Nav />
    <main>
      <Hero />
      <InAction />
      <UseCases />
      <HowItWorks />
      <OpenSource />
      <ReadmeSection />
    </main>
    <SiteFooter />
  </div>
);

export default Home;
