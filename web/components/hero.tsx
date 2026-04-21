"use client";

import { motion, useMotionValue, useSpring } from "motion/react";
import { useEffect, useRef } from "react";
import { WaveformCanvas } from "@/components/waveform-canvas";
import { CountUp } from "@/components/count-up";
import { MagneticButton } from "@/components/magnetic-button";
import { Circle, Square, Triangle } from "@/components/ui/shapes";
import { DASHBOARD_URL, DEMO_DOWNLOAD_URL, GITHUB_URL } from "@/lib/utils";

const HEADLINE = ["HEARS", "WHAT", "CCTV", "CAN'T."];

export function Hero() {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 80, damping: 14 });
  const sy = useSpring(my, { stiffness: 80, damping: 14 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      mx.set(e.clientX - r.left);
      my.set(e.clientY - r.top);
    };
    el.addEventListener("mousemove", onMove);
    return () => el.removeEventListener("mousemove", onMove);
  }, [mx, my]);

  return (
    <section
      id="top"
      ref={ref}
      className="relative border-b-6 border-ink overflow-hidden"
    >
      {/* Cursor-follow yellow disc */}
      <motion.div
        style={{ x: sx, y: sy, translateX: "-50%", translateY: "-50%" }}
        className="pointer-events-none absolute z-0 w-[380px] h-[380px] rounded-full bg-bauhaus-yellow opacity-80 mix-blend-multiply blur-[2px] hidden md:block"
      />
      {/* Slow rotating ring behind headline */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
        className="pointer-events-none absolute -left-40 top-24 z-0 hidden md:block"
      >
        <Circle className="w-[520px] h-[520px]" fill="#ffd100" />
      </motion.div>

      <div className="relative z-10 mx-auto max-w-[1400px] px-4 md:px-8 pt-16 pb-10 md:pt-24 md:pb-16">
        <div className="grid grid-cols-12 gap-4 md:gap-6">
          {/* Vertical tag */}
          <div className="hidden md:flex col-span-1 items-start justify-start">
            <span className="text-vertical font-mono text-[11px] uppercase tracking-widest opacity-70">
              SafeCommute AI · v2 · Bocconi
            </span>
          </div>

          {/* Headline block */}
          <div className="col-span-12 md:col-span-7">
            <div className="flex items-baseline gap-3 mb-4 font-mono text-[11px] uppercase tracking-widest">
              <span className="inline-block h-2 w-2 bg-bauhaus-red" />
              <span>Edge audio · Public safety</span>
            </div>
            <h1 className="font-display leading-[0.85] uppercase tracking-tight">
              {HEADLINE.map((w, i) => (
                <motion.span
                  key={w}
                  initial={{ y: "120%", rotate: -4, opacity: 0 }}
                  animate={{ y: 0, rotate: 0, opacity: 1 }}
                  transition={{
                    delay: 0.15 + i * 0.08,
                    type: "spring",
                    stiffness: 140,
                    damping: 18,
                  }}
                  className="inline-block mr-4 md:mr-6 text-mega"
                  style={i === 1 ? { color: "var(--bauhaus-red)" } : undefined}
                >
                  {w}
                </motion.span>
              ))}
            </h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="mt-6 max-w-xl font-body text-lg md:text-xl leading-snug"
            >
              Privacy-first edge audio classification for public transit, schools, and retail.{" "}
              <strong className="font-grotesk font-bold">7 MB</strong> ·{" "}
              <strong className="font-grotesk font-bold">2.8 ms</strong> INT8 ONNX ·{" "}
              <strong className="font-grotesk font-bold">no cloud</strong>.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.85, duration: 0.6 }}
              className="mt-8 flex flex-wrap gap-3"
            >
              <MagneticButton href={DEMO_DOWNLOAD_URL} variant="primary">
                Download demo ↓
              </MagneticButton>
              <MagneticButton href={DASHBOARD_URL} variant="ghost">
                Open dashboard →
              </MagneticButton>
              <MagneticButton href={GITHUB_URL} variant="outline" external>
                View on GitHub
              </MagneticButton>
            </motion.div>
          </div>

          {/* Stat discs cluster */}
          <div className="col-span-12 md:col-span-4 mt-6 md:mt-0">
            <div className="grid grid-cols-2 gap-4">
              <StatDisc
                color="bg-bauhaus-yellow"
                shape={<Circle className="absolute inset-0 w-full h-full" fill="#ffd100" />}
                big={<CountUp value={7} suffix=" MB" />}
                label="FP32 footprint"
              />
              <StatDisc
                color="bg-bauhaus-blue text-paper"
                shape={<Square className="absolute inset-0 w-full h-full" fill="#1d3bc1" />}
                big={<CountUp value={2.8} decimals={1} suffix=" ms" />}
                label="INT8 ONNX · 8T CPU"
              />
              <StatDisc
                color="bg-paper"
                shape={<Triangle className="absolute inset-0 w-full h-full" fill="#e63946" />}
                big={<CountUp value={0.804} decimals={3} />}
                label="AUC-ROC · base"
                darkText
              />
              <StatDisc
                color="bg-ink text-paper"
                big={<CountUp value={3.72} decimals={2} suffix=" MB" />}
                label="INT8 ONNX size"
              />
            </div>
            <div className="mt-3 font-mono text-[10px] uppercase tracking-widest opacity-70 leading-snug">
              Latency: Ryzen 7 7435HS, 8T, onnxruntime 1.24.4. Historical Ryzen-5 figure in /performance.
            </div>
          </div>
        </div>
      </div>

      {/* Waveform strip */}
      <div className="relative border-t-3 border-ink bg-paper">
        <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-4 flex items-center gap-4">
          <span className="font-mono text-[11px] uppercase tracking-widest whitespace-nowrap">
            Live acoustic stream ·
          </span>
          <div className="relative flex-1 h-16 md:h-20 border-3 border-ink">
            <WaveformCanvas className="absolute inset-0 w-full h-full" />
          </div>
          <span className="hidden md:inline font-mono text-[11px] uppercase tracking-widest opacity-70">
            PCEN · 64 × 188
          </span>
        </div>
      </div>
    </section>
  );
}

function StatDisc({
  color,
  shape,
  big,
  label,
  darkText,
}: {
  color: string;
  shape?: React.ReactNode;
  big: React.ReactNode;
  label: string;
  darkText?: boolean;
}) {
  return (
    <motion.div
      initial={{ scale: 0.6, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 180, damping: 16, delay: 0.3 }}
      className={`relative aspect-square border-3 border-ink ${color} ${darkText ? "text-ink" : ""} p-4 overflow-hidden`}
    >
      <div className="relative z-10 h-full flex flex-col justify-between">
        <div className="font-display text-3xl md:text-5xl leading-none tracking-tight">{big}</div>
        <div className="font-mono text-[10px] uppercase tracking-widest">{label}</div>
      </div>
    </motion.div>
  );
}
