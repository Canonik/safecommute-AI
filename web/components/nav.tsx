"use client";

import { motion } from "motion/react";
import { MagneticButton } from "@/components/magnetic-button";
import { GITHUB_URL, MAILTO_PILOT } from "@/lib/utils";

const LINKS = [
  { href: "#how", label: "How it works" },
  { href: "#performance", label: "Performance" },
  { href: "#pricing", label: "Pricing" },
  { href: "#team", label: "Team" },
];

export function Nav() {
  return (
    <motion.header
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="sticky top-0 z-40 w-full border-b-3 border-ink bg-paper/90 backdrop-blur"
    >
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-4 md:px-8 py-3">
        <a href="#top" className="flex items-center gap-2 font-display uppercase text-lg tracking-tight">
          <span className="inline-block h-4 w-4 bg-bauhaus-red" />
          <span className="inline-block h-4 w-4 rounded-full bg-bauhaus-yellow" />
          <span className="inline-block h-4 w-4 bg-bauhaus-blue" style={{ clipPath: "polygon(50% 0, 100% 100%, 0 100%)" }} />
          <span className="ml-2">SafeCommute<span className="text-bauhaus-red">.</span>AI</span>
        </a>
        <nav className="hidden md:flex items-center gap-6 font-grotesk text-sm uppercase tracking-wider">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="hover:text-bauhaus-red transition-colors">
              {l.label}
            </a>
          ))}
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="hover:text-bauhaus-red transition-colors">
            GitHub
          </a>
        </nav>
        <MagneticButton href={MAILTO_PILOT} variant="primary" className="text-xs md:text-sm px-4 py-2 md:px-5 md:py-3">
          Request pilot →
        </MagneticButton>
      </div>
    </motion.header>
  );
}
