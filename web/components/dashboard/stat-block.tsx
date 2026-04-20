"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

type Props = {
  value: string;
  label: string;
  color?: "yellow" | "blue" | "red" | "paper" | "ink";
  accentBar?: boolean;
  delay?: number;
};

const COLOR: Record<NonNullable<Props["color"]>, string> = {
  yellow: "bg-bauhaus-yellow text-ink",
  blue: "bg-bauhaus-blue text-paper",
  red: "bg-bauhaus-red text-paper",
  paper: "bg-paper text-ink",
  ink: "bg-ink text-paper",
};

export function StatBlock({ value, label, color = "paper", accentBar = true, delay = 0 }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.94 }}
      whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, amount: 0.4 }}
      transition={{ type: "spring", stiffness: 140, damping: 16, delay }}
      className={cn(
        "relative border-3 border-ink p-5 md:p-6 overflow-hidden min-h-[140px]",
        COLOR[color]
      )}
    >
      {accentBar && (
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: delay + 0.2 }}
          style={{ transformOrigin: "left" }}
          className="absolute left-0 bottom-0 h-2 w-full bg-ink"
        />
      )}
      <div className="relative z-10 h-full flex flex-col justify-between gap-4">
        <div className="font-display text-4xl md:text-6xl leading-none tracking-tight">{value}</div>
        <div className="font-mono text-[10px] uppercase tracking-widest opacity-80">{label}</div>
      </div>
    </motion.div>
  );
}
