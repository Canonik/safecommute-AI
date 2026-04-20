"use client";

import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { useRef, useState } from "react";
import { MagneticButton } from "@/components/magnetic-button";
import {
  DASHBOARD_URL,
  DEMO_DOWNLOAD_URL,
  MAILTO_PILOT,
  PRICE_PER_RUN_EUR,
  PRICE_SUBSCRIPTION_EUR,
} from "@/lib/utils";

type Tier = {
  name: string;
  price: string;
  per: string;
  tag: string;
  bg: string;
  fg: string;
  featured?: boolean;
  features: string[];
  caveat: string;
  cta:
    | { kind: "link"; label: string; href: string; external?: boolean; variant: "primary" | "outline" | "ghost" }
    | { kind: "checkout"; label: string; plan: "subscription" | "per_run"; variant: "primary" | "outline" | "ghost" };
};

const TIERS: Tier[] = [
  {
    name: "Demo",
    price: "€0",
    per: "download",
    tag: "Evaluation",
    bg: "bg-paper",
    fg: "text-ink",
    features: [
      "Base model (safecommute_v2.pth, 7 MB)",
      "short.md — architecture + usage",
      "Runs on any CPU, no cloud",
      "Inference code in the public repo",
    ],
    caveat: "~72% FP on speech before fine-tuning. Evaluate, don't deploy.",
    cta: { kind: "link", label: "Download demo ↓", href: DEMO_DOWNLOAD_URL, variant: "ghost" },
  },
  {
    name: "Per run",
    price: `€${PRICE_PER_RUN_EUR}`,
    per: "one fine-tune",
    tag: "Pay as you go",
    bg: "bg-bauhaus-yellow",
    fg: "text-ink",
    features: [
      "Upload ambient clips to your site",
      "One calibrated model per purchase",
      "Threshold recommendations included",
      "No commitment",
    ],
    caveat: "Stack credits: buy 3, fine-tune 3 sites or iterate on the same one.",
    cta: { kind: "checkout", label: `Pay €${PRICE_PER_RUN_EUR} →`, plan: "per_run", variant: "primary" },
  },
  {
    name: "Subscription",
    price: `€${PRICE_SUBSCRIPTION_EUR}`,
    per: "unlimited runs, one site",
    tag: "Recommended",
    bg: "bg-bauhaus-blue",
    fg: "text-paper",
    featured: true,
    features: [
      "Unlimited fine-tune runs on one site",
      "Re-tune after mic swaps, seasonal noise, retrofits",
      "Priority queue",
      "All future threshold tooling",
    ],
    caveat: "Break-even vs per-run at 5 tunes. Most sites tune 3–6× in year one.",
    cta: { kind: "checkout", label: `Pay €${PRICE_SUBSCRIPTION_EUR} →`, plan: "subscription", variant: "ghost" },
  },
  {
    name: "Enterprise",
    price: "Custom",
    per: "fleet & SLA",
    tag: "50+ sites",
    bg: "bg-ink",
    fg: "text-paper",
    features: [
      "Bulk-calibration API",
      "Custom unsafe-class definitions per vertical",
      "Centralized fleet dashboard",
      "Priority support + SLA",
    ],
    caveat: "For multi-vertical rollouts and compliance reporting.",
    cta: { kind: "link", label: "Contact sales", href: MAILTO_PILOT, variant: "outline" },
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
            Pay per run for one-off calibration, or subscribe for unlimited runs on a single site —
            renovations, new rolling stock, mic replacements all trigger a re-run.
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
          {TIERS.map((t, i) => (
            <TierCard key={t.name} tier={t} delay={i * 0.06} />
          ))}
        </div>

        <div className="mt-10 border-3 border-ink p-4 md:p-5 flex flex-wrap items-center gap-4 font-mono text-[11px] uppercase tracking-widest bg-paper">
          <span className="inline-block h-3 w-3 bg-bauhaus-red" />
          <span>Already got a credit?</span>
          <a
            href={DASHBOARD_URL}
            className="ml-auto border-3 border-ink bg-ink text-paper px-3 py-1.5 font-display uppercase text-xs tracking-tight hover:bg-bauhaus-red transition-colors"
          >
            Open dashboard →
          </a>
        </div>
      </div>
    </section>
  );
}

function TierCard({ tier, delay }: { tier: Tier; delay: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 120, damping: 18 });
  const sy = useSpring(my, { stiffness: 120, damping: 18 });
  const rotX = useTransform(sy, [-0.5, 0.5], [6, -6]);
  const rotY = useTransform(sx, [-0.5, 0.5], [-6, 6]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const clickCheckout = async (plan: "subscription" | "per_run") => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: plan }),
      });
      if (res.status === 401) {
        window.location.href = `/sign-in?next=${encodeURIComponent("/dashboard/billing")}`;
        return;
      }
      if (!res.ok) {
        const j = await res.json().catch(() => ({ error: "checkout failed" }));
        throw new Error(j.error ?? "checkout failed");
      }
      const { url } = await res.json();
      if (url) window.location.href = url;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
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
      className={`relative border-3 border-ink p-6 md:p-7 ${tier.bg} ${tier.fg} ${
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
      <div className="font-display uppercase text-3xl md:text-4xl mt-1 leading-none">{tier.name}</div>
      <div className="mt-5 flex items-baseline gap-2">
        <div className="font-display text-4xl md:text-5xl leading-none tracking-tight">{tier.price}</div>
        <div className="font-mono text-[11px] uppercase tracking-widest opacity-80">{tier.per}</div>
      </div>

      <ul className="mt-5 space-y-2 font-body text-sm">
        {tier.features.map((f) => (
          <li key={f} className="flex items-start gap-2">
            <span className="inline-block mt-2 h-1.5 w-1.5 bg-current opacity-80 flex-shrink-0" />
            <span>{f}</span>
          </li>
        ))}
      </ul>

      <div className="mt-5 pt-3 border-t-3 border-current opacity-90 font-body text-sm italic">{tier.caveat}</div>

      <div className="mt-5">
        {tier.cta.kind === "link" ? (
          <MagneticButton
            href={tier.cta.href}
            variant={tier.cta.variant}
            external={tier.cta.external}
            className="w-full justify-center"
          >
            {tier.cta.label}
          </MagneticButton>
        ) : (
          <MagneticButton
            onClick={() => clickCheckout(tier.cta.kind === "checkout" ? tier.cta.plan : "per_run")}
            variant={tier.cta.variant}
            className="w-full justify-center"
          >
            {loading ? "Opening…" : tier.cta.label}
          </MagneticButton>
        )}
        {error && (
          <div className="mt-3 font-mono text-[11px] uppercase tracking-widest text-bauhaus-red">
            {error}
          </div>
        )}
      </div>
    </motion.div>
  );
}
