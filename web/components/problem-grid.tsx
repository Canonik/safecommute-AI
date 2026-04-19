"use client";

import { motion } from "motion/react";
import { Square } from "@/components/ui/shapes";

const CARDS = [
  {
    num: "01",
    title: "CCTV is blind to audio.",
    body: "Video captures what already happened. The early acoustic cues — raised voices, screams, scuffles — go unmonitored.",
    color: "bg-paper",
    accent: "#e63946",
  },
  {
    num: "02",
    title: "Cloud analytics leak raw audio.",
    body: "Uploading audio for inference is a GDPR liability. Speech and biometrics cross network boundaries you don't control.",
    color: "bg-bauhaus-yellow",
    accent: "#0a0a0a",
  },
  {
    num: "03",
    title: "Universal classifiers can't adapt.",
    body: "Off-the-shelf models hit ~72% false positives on speech. Every site's ambient signature is different.",
    color: "bg-bauhaus-blue text-paper",
    accent: "#ffd100",
  },
];

export function ProblemGrid() {
  return (
    <section className="relative border-b-6 border-ink bg-paper">
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-8">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <Square className="w-3 h-3" fill="#e63946" />
              <span>The problem</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              Three gaps <br />
              <span className="text-bauhaus-red">in public-space safety.</span>
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          {CARDS.map((c, i) => (
            <motion.div
              key={c.num}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ delay: i * 0.1, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className={`relative border-3 border-ink p-6 md:p-8 min-h-[320px] flex flex-col justify-between ${c.color} overflow-hidden`}
            >
              <div
                className="absolute -right-6 -top-6 w-24 h-24"
                style={{ background: c.accent }}
              />
              <div className="relative z-10">
                <div className="font-display text-6xl md:text-7xl leading-none">{c.num}</div>
              </div>
              <div className="relative z-10">
                <h3 className="font-display text-2xl md:text-3xl leading-tight uppercase tracking-tight mb-3">
                  {c.title}
                </h3>
                <p className="font-body text-base leading-snug opacity-90">{c.body}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
