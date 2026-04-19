import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { ProblemGrid } from "@/components/problem-grid";
import { HowItWorks } from "@/components/how-it-works";
import { HonestyBlock } from "@/components/honesty-block";
import { EdgePositioning } from "@/components/edge-positioning";
import { PrivacySection } from "@/components/privacy-section";
import { VerticalsGrid } from "@/components/verticals-grid";
import { Pricing } from "@/components/pricing";
import { TeamSection } from "@/components/team-section";
import { SiteFooter } from "@/components/site-footer";
import { LiveIndicator } from "@/components/live-indicator";

export default function Home() {
  return (
    <main className="grain">
      <Nav />
      <Hero />
      <ProblemGrid />
      <HowItWorks />
      <HonestyBlock />
      <EdgePositioning />
      <PrivacySection />
      <VerticalsGrid />
      <Pricing />
      <TeamSection />
      <SiteFooter />
      <LiveIndicator />
    </main>
  );
}
