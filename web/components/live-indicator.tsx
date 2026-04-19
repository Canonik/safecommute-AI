"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Circle, Square, Triangle } from "@/components/ui/shapes";

const STATES = [
  { label: "SAFE", bg: "#1d3bc1", fg: "#f4f1ea", Shape: Circle, shapeFill: "#f4f1ea" },
  { label: "WARNING", bg: "#ffd100", fg: "#0a0a0a", Shape: Triangle, shapeFill: "#0a0a0a" },
  { label: "ALERT", bg: "#e63946", fg: "#f4f1ea", Shape: Square, shapeFill: "#f4f1ea" },
] as const;

export function LiveIndicator() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((n) => (n + 1) % STATES.length), 2800);
    return () => clearInterval(id);
  }, []);
  const s = STATES[i];
  return (
    <div className="fixed bottom-5 right-5 z-50 select-none">
      <AnimatePresence mode="wait">
        <motion.div
          key={s.label}
          initial={{ y: 10, opacity: 0, scale: 0.95 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          exit={{ y: -6, opacity: 0 }}
          transition={{ type: "spring", stiffness: 240, damping: 22 }}
          className="flex items-center gap-3 border-3 border-ink px-4 py-2 font-display text-sm uppercase tracking-tight shadow-[6px_6px_0_0_#0a0a0a]"
          style={{ background: s.bg, color: s.fg }}
        >
          <motion.span
            animate={{ scale: [1, 1.15, 1] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
            className="inline-block"
          >
            <s.Shape className="w-3.5 h-3.5" fill={s.shapeFill} />
          </motion.span>
          <span>{s.label}</span>
        </motion.div>
      </AnimatePresence>
      <div className="mt-1 text-right font-mono text-[10px] uppercase tracking-widest text-ink/60">
        live demo · simulated
      </div>
    </div>
  );
}
