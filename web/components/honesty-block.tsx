"use client";

import { motion, useInView } from "motion/react";
import { useRef } from "react";

const THREATS = [
  { name: "as_yell", acc: 90.6 },
  { name: "as_screaming", acc: 79.1 },
  { name: "yt_scream", acc: 78.2 },
  { name: "as_shout", acc: 64.7 },
];
const HARD = [
  { name: "yt_metro", acc: 64.9 },
  { name: "as_crowd", acc: 42.1 },
  { name: "as_speech", acc: 28.3 },
  { name: "as_laughter", acc: 17.5 },
];

function Row({ name, acc, color, delay, reverse }: { name: string; acc: number; color: string; delay: number; reverse?: boolean }) {
  return (
    <div className="flex items-center gap-3 font-mono text-xs md:text-sm">
      <span className="w-28 md:w-32 opacity-80">{name}</span>
      <div className="flex-1 h-5 border-3 border-ink relative overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${acc}%` }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ delay, duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="h-full"
          style={{ background: color }}
        />
      </div>
      <span className="w-12 text-right">{acc.toFixed(1)}%</span>
    </div>
  );
}

export function HonestyBlock() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.4 });

  return (
    <section id="performance" className="relative border-b-6 border-ink bg-paper">
      <span className="hidden md:block absolute right-3 top-24 text-vertical font-mono text-[11px] uppercase tracking-widest opacity-60">
        Performance · honest numbers
      </span>

      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <span className="inline-block h-3 w-3 bg-bauhaus-blue" />
              <span>Honesty block</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              What the base <br /> model does — <br />
              <span className="text-bauhaus-red">and doesn&apos;t.</span>
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 md:col-span-7 border-3 border-ink p-6 bg-paper">
            <div className="font-display uppercase text-xl mb-4">Per-source accuracy (base v2)</div>
            <div className="font-mono text-[11px] uppercase tracking-widest mb-2 text-bauhaus-blue">Threats — detects well</div>
            <div className="space-y-2 mb-5">
              {THREATS.map((r, i) => (
                <Row key={r.name} {...r} color="#1d3bc1" delay={0.1 + i * 0.08} />
              ))}
            </div>
            <div className="font-mono text-[11px] uppercase tracking-widest mb-2 text-bauhaus-red">Hard negatives — struggles</div>
            <div className="space-y-2">
              {HARD.map((r, i) => (
                <Row key={r.name} {...r} color="#e63946" delay={0.4 + i * 0.08} />
              ))}
            </div>
            <div className="mt-5 pt-4 border-t-3 border-ink font-body text-sm">
              The base model confuses speech with shouting. Out of the box, it is a demo — not a deployable product.
              That&apos;s the whole point of the fine-tuning step.
            </div>
          </div>

          <div className="col-span-12 md:col-span-5 flex flex-col gap-6">
            <motion.div
              ref={ref}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.6 }}
              className="relative border-3 border-ink bg-bauhaus-yellow p-6 overflow-hidden"
            >
              <div className="font-mono text-[11px] uppercase tracking-widest mb-4">
                After site calibration (measured, n=1 site)
              </div>
              <div className="font-display leading-none tracking-tight">
                <div className="text-6xl md:text-7xl mb-1">
                  <span className="line-through decoration-[6px] decoration-ink opacity-60">72%</span>
                </div>
                <div className="text-7xl md:text-8xl text-bauhaus-red">
                  ~15%
                </div>
                <div className="mt-2 text-base uppercase tracking-tight">speech false-positive rate</div>
              </div>
              <div className="mt-4 font-body text-sm">
                Frozen CNN + GRU/FC adaptation on 30+ min of recorded ambient. Threat recall 89.5%.
                Overall-FP plateau is 29–38%; the ≤5% deployment gate is not yet met on n=1 site.
                Full breakdown in the paper.
              </div>

              {/* Shrinking bar */}
              <div className="mt-6">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest mb-1">
                  <span>Speech FP rate</span>
                </div>
                <div className="relative h-7 border-3 border-ink bg-paper overflow-hidden">
                  <motion.div
                    initial={{ width: "72%" }}
                    animate={inView ? { width: "15%" } : { width: "72%" }}
                    transition={{ duration: 1.4, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
                    className="absolute inset-y-0 left-0 bg-bauhaus-red"
                  />
                </div>
              </div>
            </motion.div>

            <div className="relative border-3 border-ink bg-paper p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/figures/finetune_impact.png"
                alt="Fine-tuning impact — measured deployment result on metro (n=1 site)"
                className="w-full h-auto"
              />
              <div className="mt-2 px-2 pb-1 font-mono text-[11px] uppercase tracking-widest opacity-70">
                Fig · Fine-tune impact
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
