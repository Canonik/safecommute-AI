"use client";

import { motion } from "motion/react";

const STAGES = [
  { label: "Mic", sub: "16 kHz PCM", color: "#0a0a0a", text: "#f4f1ea" },
  { label: "Mel", sub: "64 bands", color: "#0a0a0a", text: "#f4f1ea" },
  { label: "PCEN", sub: "non-invertible", color: "#e63946", text: "#f4f1ea", boundary: true },
  { label: "Tensor", sub: "(1,64,188)", color: "#1d3bc1", text: "#f4f1ea" },
  { label: "CNN+GRU", sub: "classifier", color: "#ffd100", text: "#0a0a0a" },
];

export function PrivacySection() {
  return (
    <section className="relative border-b-6 border-ink bg-paper">
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 mb-10">
          <div className="col-span-12 md:col-span-8">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <span className="inline-block h-3 w-3 bg-bauhaus-red" />
              <span>Privacy architecture</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              No audio <br /> ever <span className="text-bauhaus-red">leaves</span> <br /> the device.
            </h2>
          </div>
          <div className="col-span-12 md:col-span-4 font-body text-base leading-snug md:pt-16">
            <p className="mb-3">
              PCEN is lossy by construction. The raw waveform is <strong>mathematically unrecoverable</strong> from the
              spectrogram tiles the classifier consumes.
            </p>
            <p className="text-ink/70">
              This is a structural guarantee — not a policy promise. There is no upload, no retention,
              no human review path.
            </p>
          </div>
        </div>

        {/* Signal flow */}
        <div className="border-3 border-ink bg-ink text-paper p-6 md:p-10 relative overflow-hidden">
          <div className="font-mono text-[11px] uppercase tracking-widest opacity-70 mb-6">
            Signal flow · the red boundary is where waveform is destroyed
          </div>
          <div className="grid grid-cols-5 gap-2 md:gap-4 items-stretch">
            {STAGES.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                className="relative border-3 border-paper p-3 md:p-5 flex flex-col items-start justify-between min-h-[140px]"
                style={{ background: s.color, color: s.text, borderColor: s.boundary ? "#ffd100" : "#f4f1ea" }}
              >
                <div className="font-mono text-[10px] uppercase tracking-widest opacity-80">Stage {i + 1}</div>
                <div>
                  <div className="font-display text-xl md:text-2xl uppercase leading-none">{s.label}</div>
                  <div className="font-mono text-[11px] opacity-80 mt-1">{s.sub}</div>
                </div>
                {i < STAGES.length - 1 && (
                  <div className="hidden md:block absolute -right-4 top-1/2 -translate-y-1/2 z-10">
                    <motion.div
                      animate={{ x: [0, 4, 0] }}
                      transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                      className="w-8 h-0.5 bg-paper relative"
                    >
                      <span
                        className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-y-[5px] border-y-transparent border-l-[8px]"
                        style={{ borderLeftColor: "#f4f1ea" }}
                      />
                    </motion.div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* boundary label */}
          <div className="relative mt-6 flex">
            <div className="hidden md:block" style={{ width: "40%" }} />
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="flex items-center gap-3 px-3 py-1 bg-bauhaus-yellow text-ink font-mono text-[11px] uppercase tracking-widest border-3 border-paper"
            >
              <span className="inline-block w-2 h-2 bg-bauhaus-red" />
              waveform destroyed — audio is no longer reconstructible beyond this line
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
