"use client";

import Link from "next/link";
import { motion } from "motion/react";
import { Circle, Square, Triangle } from "@/components/ui/shapes";

type Site = { id: string; name: string; environment: string; created_at: string };

const ENV_META: Record<string, { color: string; Shape: typeof Circle; fg: string }> = {
  metro: { color: "bg-bauhaus-blue", Shape: Square, fg: "text-paper" },
  retail: { color: "bg-bauhaus-yellow", Shape: Circle, fg: "text-ink" },
  school: { color: "bg-bauhaus-red", Shape: Triangle, fg: "text-paper" },
  eldercare: { color: "bg-paper", Shape: Circle, fg: "text-ink" },
  industrial: { color: "bg-ink", Shape: Square, fg: "text-paper" },
  other: { color: "bg-paper", Shape: Triangle, fg: "text-ink" },
};

export function SiteGrid({ sites }: { sites: Site[] }) {
  if (!sites.length) {
    return (
      <div className="border-3 border-ink border-dashed p-10 md:p-14 text-center">
        <div className="font-display uppercase text-2xl mb-2">No sites yet</div>
        <div className="font-grotesk text-sm opacity-80">
          Create one to start uploading ambient recordings.
        </div>
        <Link
          href="/dashboard/sites/new"
          className="mt-6 inline-flex border-3 border-ink bg-ink text-paper px-5 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red transition-colors"
        >
          + Create first site
        </Link>
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
      {sites.map((s, i) => {
        const meta = ENV_META[s.environment] ?? ENV_META.other;
        const Shape = meta.Shape;
        return (
          <motion.div
            key={s.id}
            initial={{ opacity: 0, y: 24, rotate: -1 }}
            animate={{ opacity: 1, y: 0, rotate: 0 }}
            transition={{ type: "spring", stiffness: 140, damping: 18, delay: i * 0.05 }}
            whileHover={{ y: -4, rotate: 0.5 }}
          >
            <Link
              href={`/dashboard/sites/${s.id}`}
              className={`relative block border-3 border-ink ${meta.color} ${meta.fg} p-5 md:p-6 shadow-[6px_6px_0_0_#0a0a0a] hover:shadow-[10px_10px_0_0_#0a0a0a] transition-shadow overflow-hidden`}
            >
              <div className="absolute -top-6 -right-6 opacity-60">
                <Shape className="w-24 h-24" fill="currentColor" />
              </div>
              <div className="relative z-10">
                <div className="font-mono text-[10px] uppercase tracking-widest opacity-80">
                  {s.environment}
                </div>
                <div className="font-display uppercase text-2xl md:text-3xl leading-[0.95] tracking-tight mt-1">
                  {s.name}
                </div>
                <div className="font-mono text-[10px] uppercase tracking-widest opacity-70 mt-8">
                  created {new Date(s.created_at).toLocaleDateString()}
                </div>
              </div>
            </Link>
          </motion.div>
        );
      })}
    </div>
  );
}
