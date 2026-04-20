"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";

type Props = {
  eyebrow: string;
  title: string;
  accent?: "red" | "blue" | "yellow";
  children?: ReactNode;
};

const ACCENT = {
  red: "bg-bauhaus-red",
  blue: "bg-bauhaus-blue",
  yellow: "bg-bauhaus-yellow",
} as const;

export function PageHeading({ eyebrow, title, accent = "red", children }: Props) {
  return (
    <div className="grid grid-cols-12 gap-6 items-end mb-8 md:mb-12">
      <div className="col-span-12 md:col-span-9">
        <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
          <span className={`inline-block h-3 w-3 ${ACCENT[accent]}`} />
          <span>{eyebrow}</span>
        </div>
        <motion.h1
          initial={{ y: 40, opacity: 0, rotate: -2 }}
          animate={{ y: 0, opacity: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 140, damping: 18 }}
          className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight"
        >
          {title}
        </motion.h1>
      </div>
      {children && <div className="col-span-12 md:col-span-3 flex md:justify-end">{children}</div>}
    </div>
  );
}
