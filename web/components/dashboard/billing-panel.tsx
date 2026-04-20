"use client";

import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { useRef, useState } from "react";
import { MagneticButton } from "@/components/magnetic-button";
import { Circle, Square, Triangle } from "@/components/ui/shapes";
import { PRICE_PER_RUN_EUR, PRICE_SUBSCRIPTION_EUR } from "@/lib/utils";

type Payment = {
  kind: "subscription" | "per_run";
  amount_cents: number;
  currency: string;
  status: string;
  created_at: string;
};

export function BillingPanel({
  subscriptionActive,
  perRunCredits,
  payments,
  resultStatus,
}: {
  subscriptionActive: boolean;
  perRunCredits: number;
  payments: Payment[];
  resultStatus: string | null;
}) {
  const [loading, setLoading] = useState<"subscription" | "per_run" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkout = async (kind: "subscription" | "per_run") => {
    setLoading(kind);
    setError(null);
    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({ error: "checkout failed" }));
        throw new Error(j.error ?? "checkout failed");
      }
      const { url } = await res.json();
      if (url) window.location.href = url;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="space-y-8">
      {resultStatus === "success" && (
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="border-3 border-ink bg-bauhaus-blue text-paper p-4 font-display uppercase text-sm tracking-tight"
        >
          Payment received — credit applied.
        </motion.div>
      )}
      {resultStatus === "cancel" && (
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="border-3 border-ink bg-bauhaus-yellow p-4 font-display uppercase text-sm tracking-tight"
        >
          Checkout cancelled. No charge made.
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <TierCard
          tag="Unlimited runs"
          name="Subscription"
          price={`€${PRICE_SUBSCRIPTION_EUR}`}
          per="one-off, this site"
          bg="bg-bauhaus-blue"
          fg="text-paper"
          Shape={Square}
          features={[
            "Unlimited fine-tune runs",
            "Re-tune after mic or venue changes",
            "Priority queue when the worker is busy",
          ]}
          cta={
            <MagneticButton
              onClick={() => checkout("subscription")}
              variant="ghost"
              className="w-full justify-center"
            >
              {loading === "subscription" ? "Opening checkout…" : `Pay €${PRICE_SUBSCRIPTION_EUR} →`}
            </MagneticButton>
          }
          active={subscriptionActive}
          activeLabel="Active"
        />
        <TierCard
          tag="Pay per run"
          name="Single run"
          price={`€${PRICE_PER_RUN_EUR}`}
          per="per fine-tune"
          bg="bg-bauhaus-yellow"
          fg="text-ink"
          Shape={Circle}
          features={[
            "One calibrated model per purchase",
            "No commitment",
            "Stacks — buy multiple credits upfront",
          ]}
          cta={
            <MagneticButton
              onClick={() => checkout("per_run")}
              variant="primary"
              className="w-full justify-center"
            >
              {loading === "per_run" ? "Opening checkout…" : `Pay €${PRICE_PER_RUN_EUR} →`}
            </MagneticButton>
          }
          active={perRunCredits > 0}
          activeLabel={`${perRunCredits} credit${perRunCredits === 1 ? "" : "s"}`}
          featured
        />
      </div>

      {error && (
        <div className="border-3 border-bauhaus-red bg-bauhaus-red/10 px-3 py-2 font-mono text-[11px]">
          {error}
        </div>
      )}

      <div>
        <div className="mb-3 flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest">
          <Triangle className="w-3 h-3" fill="#0a0a0a" />
          <span>History</span>
        </div>
        <div className="border-3 border-ink divide-y divide-ink/30">
          {payments.length === 0 && (
            <div className="p-4 font-grotesk text-sm opacity-70">No payments yet.</div>
          )}
          {payments.map((p, i) => (
            <div key={i} className="flex items-center gap-4 p-3 md:p-4 font-mono text-[11px] uppercase tracking-widest">
              <span className="opacity-60 w-28">{new Date(p.created_at).toLocaleDateString()}</span>
              <span className="flex-1 font-display">{p.kind === "subscription" ? "Subscription" : "Per-run"}</span>
              <span>
                {(p.amount_cents / 100).toFixed(2)} {p.currency.toUpperCase()}
              </span>
              <span className="bg-ink text-paper px-2 py-0.5">{p.status}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TierCard({
  tag,
  name,
  price,
  per,
  bg,
  fg,
  features,
  cta,
  Shape,
  active,
  activeLabel,
  featured = false,
}: {
  tag: string;
  name: string;
  price: string;
  per: string;
  bg: string;
  fg: string;
  features: string[];
  cta: React.ReactNode;
  Shape: typeof Circle;
  active: boolean;
  activeLabel: string;
  featured?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 120, damping: 18 });
  const sy = useSpring(my, { stiffness: 120, damping: 18 });
  const rotX = useTransform(sy, [-0.5, 0.5], [6, -6]);
  const rotY = useTransform(sx, [-0.5, 0.5], [-6, 6]);

  return (
    <motion.div
      ref={ref}
      onMouseMove={(e) => {
        const r = ref.current?.getBoundingClientRect();
        if (!r) return;
        mx.set((e.clientX - r.left) / r.width - 0.5);
        my.set((e.clientY - r.top) / r.height - 0.5);
      }}
      onMouseLeave={() => {
        mx.set(0);
        my.set(0);
      }}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.6 }}
      style={{ rotateX: rotX, rotateY: rotY, transformStyle: "preserve-3d", transformPerspective: 1000 }}
      className={`relative border-3 border-ink p-6 md:p-8 ${bg} ${fg} overflow-hidden ${
        featured ? "shadow-[8px_8px_0_0_#0a0a0a]" : ""
      }`}
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
        className="absolute -top-12 -right-12 pointer-events-none"
      >
        <Shape className="w-40 h-40 opacity-40" fill="currentColor" />
      </motion.div>
      {active && (
        <div className="absolute -top-3 left-6 border-3 border-ink bg-ink text-paper px-3 py-1 font-display uppercase text-xs tracking-widest">
          {activeLabel}
        </div>
      )}
      <div className="relative z-10">
        <div className="font-mono text-[11px] uppercase tracking-widest opacity-80">{tag}</div>
        <div className="font-display uppercase text-4xl md:text-5xl mt-1 leading-none">{name}</div>
        <div className="mt-6 flex items-baseline gap-2">
          <div className="font-display text-5xl md:text-6xl leading-none tracking-tight">{price}</div>
          <div className="font-mono text-[11px] uppercase tracking-widest opacity-80">{per}</div>
        </div>
        <ul className="mt-6 space-y-2 font-grotesk text-sm">
          {features.map((f) => (
            <li key={f} className="flex items-start gap-2">
              <span className="inline-block mt-2 h-1.5 w-1.5 bg-current opacity-80 flex-shrink-0" />
              <span>{f}</span>
            </li>
          ))}
        </ul>
        <div className="mt-8">{cta}</div>
      </div>
    </motion.div>
  );
}
