"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { Circle, Square, Triangle } from "@/components/ui/shapes";

export function DashboardUnavailable() {
  return (
    <div className="min-h-screen bg-paper text-ink flex items-center justify-center px-4 py-16">
      <div className="relative max-w-xl w-full border-6 border-ink bg-paper p-8 md:p-12 shadow-[10px_10px_0_0_#0a0a0a]">
        <div className="absolute -top-6 -left-6">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 24, repeat: Infinity, ease: "linear" }}
          >
            <Circle className="w-24 h-24" fill="#ffd100" />
          </motion.div>
        </div>
        <div className="absolute -bottom-8 -right-8">
          <Square className="w-32 h-32" fill="#1d3bc1" />
        </div>
        <div className="relative z-10">
          <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
            <Triangle className="w-3 h-3" fill="#e63946" />
            <span>Not configured</span>
          </div>
          <h1 className="font-display uppercase text-4xl md:text-5xl leading-[0.9] tracking-tight">
            Dashboard offline
          </h1>
          <p className="font-grotesk mt-6 text-base leading-snug">
            The dashboard backend (Supabase + Stripe) has not been wired up on this deployment.
            Check <code className="font-mono">DEPLOY_WEB.md</code> for the five-minute setup.
          </p>
          <div className="mt-8 flex gap-3 flex-wrap">
            <Link
              href="/"
              className="border-3 border-ink px-4 py-3 font-display uppercase text-sm tracking-tight bg-ink text-paper hover:bg-bauhaus-red transition-colors"
            >
              Back to site
            </Link>
            <a
              href="/demo/safecommute-v2-demo.zip"
              className="border-3 border-ink px-4 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-yellow transition-colors"
            >
              Download demo ↓
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
