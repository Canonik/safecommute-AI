"use client";

import { motion } from "motion/react";
import { Triangle } from "@/components/ui/shapes";

const STEPS = [
  { n: "01", t: "Layered data", d: "Universal threats · hard negatives · per-site ambient. Source-level sha256 splits — no leakage." },
  { n: "02", t: "PCEN features", d: "16 kHz → 64-band mel → PCEN. Non-invertible by design: waveform is destroyed at the feature stage." },
  { n: "03", t: "Base training", d: "Focal γ=0.5 + 30% noise injection. 1.83 M params, 7 MB float32, ~12 ms CPU." },
  { n: "04", t: "Site fine-tune", d: "10 minutes on recorded ambient. Frozen CNN, adapts GRU + FC. Speech FP 72% → <5%.", featured: true },
  { n: "05", t: "Edge inference", d: "3-sec sliding window, ambient calibration, speech-aware threshold, temporal smoothing." },
];

export function HowItWorks() {
  return (
    <section id="how" className="relative border-b-6 border-ink bg-ink text-paper overflow-hidden">
      {/* decorative vertical label */}
      <span className="hidden md:block absolute left-3 top-24 text-vertical font-mono text-[11px] uppercase tracking-widest opacity-60">
        Pipeline · 5 stages
      </span>

      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4 text-bauhaus-yellow">
              <Triangle className="w-3 h-3" fill="#ffd100" />
              <span>How it works</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              Base model <span className="text-bauhaus-yellow">+</span> <br />
              per-site <span className="text-bauhaus-red">fine-tune.</span>
            </h2>
          </div>
        </div>

        {/* Animated packet line */}
        <div className="relative mb-8 h-[3px] bg-paper/20 overflow-hidden">
          <motion.div
            initial={{ x: "-20%" }}
            animate={{ x: "120%" }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            className="absolute top-1/2 -translate-y-1/2 h-2 w-24"
            style={{
              background: "linear-gradient(90deg, transparent, #e63946 30%, #ffd100 55%, #1d3bc1 75%, transparent)",
            }}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 md:gap-4">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.n}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ delay: i * 0.08, duration: 0.55 }}
              className={`relative border-3 ${
                s.featured
                  ? "border-bauhaus-yellow bg-bauhaus-yellow text-ink md:row-span-1 md:-translate-y-2 md:shadow-[6px_6px_0_0_#e63946]"
                  : "border-paper bg-transparent"
              } p-5 min-h-[220px] flex flex-col justify-between`}
            >
              <div className="font-display text-4xl leading-none tracking-tight">{s.n}</div>
              <div>
                <div className="font-display uppercase text-lg leading-tight mb-2">{s.t}</div>
                <div className="font-body text-sm leading-snug opacity-90">{s.d}</div>
              </div>
              {s.featured && (
                <div className="absolute -top-3 -right-3 bg-bauhaus-red text-paper font-mono text-[10px] uppercase tracking-widest px-2 py-1 border-3 border-ink">
                  Billable
                </div>
              )}
            </motion.div>
          ))}
        </div>

        <div className="mt-10 grid grid-cols-12 gap-6 items-start">
          <div className="col-span-12 md:col-span-7">
            <div className="border-3 border-paper bg-paper p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/figures/privacy_pipeline.png"
                alt="PCEN pipeline: waveform is destroyed at the feature stage"
                className="w-full h-auto"
              />
            </div>
            <div className="mt-2 font-mono text-[11px] uppercase tracking-widest opacity-70">
              Fig · Privacy pipeline · waveform is destroyed at the PCEN stage
            </div>
          </div>
          <div className="col-span-12 md:col-span-5 font-body leading-snug">
            <h3 className="font-display uppercase text-3xl mb-3">Privacy by construction</h3>
            <p className="mb-3">
              PCEN (Per-Channel Energy Normalization) is lossy and non-invertible. Once audio crosses the PCEN stage,
              there is no audio left to reconstruct — only a non-speech spectrogram tile.
            </p>
            <p className="text-paper/70 text-sm">
              Unlike &ldquo;we promise not to store audio&rdquo; policies, reconstruction is physically impossible.
              This is the structural GDPR guarantee.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
