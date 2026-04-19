"use client";

import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { useRef } from "react";
import { MagneticButton } from "@/components/magnetic-button";
import { GITHUB_URL, MAILTO_PILOT } from "@/lib/utils";

const TIERS = [
  {
    name: "Free",
    price: "$0",
    per: "forever",
    tag: "Demo / evaluation",
    bg: "bg-bauhaus-blue",
    fg: "text-paper",
    features: [
      "Base model (safecommute_v2.pth, 7 MB)",
      "Inference runtime (CLI)",
      "Live demo script",
      "Full documentation",
    ],
    caveat: "~72% FP on speech. A working demo, not a deployment.",
    cta: { label: "View on GitHub", href: GITHUB_URL, external: true, variant: "ghost" as const },
  },
  {
    name: "Pro",
    price: "$200",
    per: "/ site / year",
    tag: "Recommended",
    bg: "bg-bauhaus-yellow",
    fg: "text-ink",
    featured: true,
    features: [
      "Self-serve calibration tool",
      "Threshold optimization (Youden's J / F1 / low-FPR)",
      "Base model updates",
      "Unlimited re-calibrations per site",
    ],
    caveat: "Operators never touch Python. Record ambient → click calibrate → deploy.",
    cta: { label: "Request pilot →", href: MAILTO_PILOT, variant: "primary" as const },
  },
  {
    name: "Enterprise",
    price: "Custom",
    per: "from ~$2K / year",
    tag: "Fleet & SLA",
    bg: "bg-ink",
    fg: "text-paper",
    features: [
      "Bulk-calibration API",
      "Custom unsafe-class definitions per vertical",
      "Centralized fleet dashboard",
      "Priority support + SLA",
    ],
    caveat: "For 50+ sites, multi-vertical rollouts, compliance reporting.",
    cta: { label: "Contact sales", href: MAILTO_PILOT, variant: "outline" as const },
  },
];

export function Pricing() {
  return (
    <section id="pricing" className="relative border-b-6 border-ink bg-paper">
      <span className="hidden md:block absolute right-3 top-24 text-vertical font-mono text-[11px] uppercase tracking-widest opacity-60">
        Pricing · per site
      </span>
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <span className="inline-block h-3 w-3 bg-bauhaus-yellow" />
              <span>Pricing</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              Give away the model. <br />
              <span className="text-bauhaus-red">Sell the calibration.</span>
            </h2>
          </div>
          <div className="col-span-12 md:col-span-3 font-body text-sm leading-snug">
            Annual subscription covers unlimited re-calibrations — renovations, new rolling stock, mic replacements all
            trigger a re-run.
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          {TIERS.map((t, i) => (
            <TierCard key={t.name} tier={t} delay={i * 0.08} />
          ))}
        </div>
      </div>
    </section>
  );
}

type Tier = (typeof TIERS)[number];

function TierCard({ tier, delay }: { tier: Tier; delay: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 120, damping: 18 });
  const sy = useSpring(my, { stiffness: 120, damping: 18 });
  const rotX = useTransform(sy, [-0.5, 0.5], [6, -6]);
  const rotY = useTransform(sx, [-0.5, 0.5], [-6, 6]);

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    mx.set((e.clientX - r.left) / r.width - 0.5);
    my.set((e.clientY - r.top) / r.height - 0.5);
  };
  const onLeave = () => {
    mx.set(0);
    my.set(0);
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ delay, duration: 0.6 }}
      style={{ rotateX: rotX, rotateY: rotY, transformStyle: "preserve-3d", transformPerspective: 1000 }}
      className={`relative border-3 border-ink p-6 md:p-8 ${tier.bg} ${tier.fg} ${
        tier.featured ? "md:-translate-y-3 shadow-[8px_8px_0_0_#0a0a0a]" : ""
      }`}
    >
      {tier.featured && (
        <motion.div
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -top-4 right-6 bg-bauhaus-red text-paper font-display uppercase text-xs tracking-widest px-3 py-1 border-3 border-ink"
        >
          Recommended
        </motion.div>
      )}
      <div className="font-mono text-[11px] uppercase tracking-widest opacity-80">{tier.tag}</div>
      <div className="font-display uppercase text-4xl md:text-5xl mt-1 leading-none">{tier.name}</div>
      <div className="mt-6 flex items-baseline gap-2">
        <div className="font-display text-5xl md:text-6xl leading-none tracking-tight">{tier.price}</div>
        <div className="font-mono text-[11px] uppercase tracking-widest opacity-80">{tier.per}</div>
      </div>

      <ul className="mt-6 space-y-2 font-body text-sm">
        {tier.features.map((f) => (
          <li key={f} className="flex items-start gap-2">
            <span className="inline-block mt-2 h-1.5 w-1.5 bg-current opacity-80 flex-shrink-0" />
            <span>{f}</span>
          </li>
        ))}
      </ul>

      <div className="mt-6 pt-4 border-t-3 border-current opacity-90 font-body text-sm italic">{tier.caveat}</div>

      <div className="mt-6">
        <MagneticButton
          href={tier.cta.href}
          variant={tier.cta.variant}
          external={tier.cta.external}
          className="w-full justify-center"
        >
          {tier.cta.label}
        </MagneticButton>
      </div>
    </motion.div>
  );
}
