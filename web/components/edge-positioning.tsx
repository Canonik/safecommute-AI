"use client";

import { motion, useScroll, useTransform } from "motion/react";
import { useRef } from "react";

const STATS = [
  { v: "1.83M", l: "Parameters" },
  { v: "7 MB", l: "Float32 size" },
  { v: "2.8 ms", l: "INT8 · 8T CPU" },
  { v: "ARM", l: "benchmark pending" },
  { v: "0 GPU", l: "Required" },
];

export function EdgePositioning() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], ["-6%", "6%"]);

  return (
    <section ref={ref} className="relative border-b-6 border-ink bg-bauhaus-blue text-paper overflow-hidden">
      <span className="hidden md:block absolute left-3 top-24 text-vertical font-mono text-[11px] uppercase tracking-widest opacity-80">
        Edge · deployable
      </span>

      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4 text-bauhaus-yellow">
              <span className="inline-block h-3 w-3 bg-bauhaus-yellow" />
              <span>Positioning</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              The only model in the <br />
              <span className="text-bauhaus-yellow">edge-deployable corner.</span>
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6 items-start">
          <div className="col-span-12 md:col-span-8 relative border-3 border-ink bg-paper overflow-hidden">
            <motion.div style={{ y }} className="p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/figures/footprint_bubble.png"
                alt="Model footprint vs latency — SafeCommute sits alone in the edge-deployable corner"
                className="w-full h-auto"
              />
            </motion.div>
            <div className="border-t-3 border-ink px-4 py-2 font-mono text-[11px] uppercase tracking-widest text-ink">
              Fig · params × latency, bubble-sized by score. Against YAMNet · CNN14 · AST.
            </div>
          </div>

          <div className="col-span-12 md:col-span-4 space-y-3">
            {STATS.map((s, i) => (
              <motion.div
                key={s.l}
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ delay: i * 0.08, duration: 0.5 }}
                className="border-3 border-paper p-4 flex items-baseline justify-between"
              >
                <span className="font-display text-3xl md:text-4xl tracking-tight">{s.v}</span>
                <span className="font-mono text-[11px] uppercase tracking-widest opacity-80">{s.l}</span>
              </motion.div>
            ))}
            <div className="border-3 border-bauhaus-yellow bg-bauhaus-yellow text-ink p-4 font-body text-sm">
              <strong className="font-display uppercase text-base block mb-1">44.7× smaller, 22.7× faster than CNN14</strong>
              Measured 2026-04-22 on Ryzen 7 7435HS, 8T. Runs on the hardware you already have at the edge.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
