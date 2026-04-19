"use client";

import { motion } from "motion/react";
import { GITHUB_URL, MAILTO_CONTACT, MAILTO_PILOT } from "@/lib/utils";
import { MagneticButton } from "@/components/magnetic-button";

export function SiteFooter() {
  return (
    <footer className="relative bg-ink text-paper overflow-hidden">
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-8">
          <div className="col-span-12 md:col-span-8">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="font-display uppercase text-6xl md:text-giga leading-[0.85] tracking-tight"
            >
              No raw audio <br />
              <span className="text-bauhaus-yellow">is ever stored.</span>
            </motion.h2>
            <p className="mt-6 font-body text-base md:text-lg max-w-xl opacity-80">
              SafeCommute AI · built at Bocconi University · base model Apache-licensed · calibration platform commercial.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <MagneticButton href={MAILTO_PILOT} variant="ghost">
                Request calibration pilot →
              </MagneticButton>
              <MagneticButton href={GITHUB_URL} variant="outline" external>
                GitHub
              </MagneticButton>
            </div>
          </div>

          <div className="col-span-6 md:col-span-2 font-mono text-[11px] uppercase tracking-widest">
            <div className="opacity-60 mb-3">Product</div>
            <ul className="space-y-2">
              <li><a href="#how" className="hover:text-bauhaus-yellow">How it works</a></li>
              <li><a href="#performance" className="hover:text-bauhaus-yellow">Performance</a></li>
              <li><a href="#pricing" className="hover:text-bauhaus-yellow">Pricing</a></li>
              <li><a href="#team" className="hover:text-bauhaus-yellow">Team</a></li>
            </ul>
          </div>

          <div className="col-span-6 md:col-span-2 font-mono text-[11px] uppercase tracking-widest">
            <div className="opacity-60 mb-3">Contact</div>
            <ul className="space-y-2">
              <li><a href={MAILTO_CONTACT} className="hover:text-bauhaus-yellow break-all">email →</a></li>
              <li><a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="hover:text-bauhaus-yellow">github →</a></li>
            </ul>
          </div>
        </div>

        <div className="mt-16 pt-6 border-t-3 border-paper/30 flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] uppercase tracking-widest opacity-70">
          <div>© 2026 SafeCommute AI</div>
          <div className="flex items-center gap-4">
            <span className="inline-block h-3 w-3 bg-bauhaus-red" />
            <span className="inline-block h-3 w-3 rounded-full bg-bauhaus-yellow" />
            <span className="inline-block h-3 w-3 bg-bauhaus-blue" style={{ clipPath: "polygon(50% 0, 100% 100%, 0 100%)" }} />
          </div>
          <div>PCEN · CNN6 · SE · GRU · edge-only</div>
        </div>
      </div>
    </footer>
  );
}
