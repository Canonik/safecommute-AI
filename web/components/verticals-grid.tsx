"use client";

import { motion } from "motion/react";
import { useState } from "react";

const VERTICALS = [
  {
    n: "01",
    name: "Public transit",
    buyer: "Transit authorities",
    unsafe: ["Screams", "Fights", "Glass breaking", "Raised voices on platform"],
    color: "bg-bauhaus-red text-paper",
  },
  {
    n: "02",
    name: "Retail",
    buyer: "Mall operators",
    unsafe: ["Aggressive shouting", "Breaking glass", "Confrontational speech"],
    color: "bg-bauhaus-yellow text-ink",
  },
  {
    n: "03",
    name: "Schools",
    buyer: "School districts",
    unsafe: ["Bullying escalation", "Distress calls", "Corridor fights"],
    color: "bg-bauhaus-blue text-paper",
  },
  {
    n: "04",
    name: "Elder care",
    buyer: "Care facilities",
    unsafe: ["Falls", "Distress calls", "Alarm sounds"],
    color: "bg-paper text-ink",
  },
  {
    n: "05",
    name: "Industrial",
    buyer: "Factories",
    unsafe: ["Machine faults", "Alarms", "Shouted warnings"],
    color: "bg-ink text-paper",
  },
];

export function VerticalsGrid() {
  return (
    <section className="relative border-b-6 border-ink bg-paper">
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <span className="inline-block h-3 w-3 bg-bauhaus-blue" />
              <span>Verticals</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              One pipeline. <br />
              <span className="text-bauhaus-blue">Five markets.</span>
            </h2>
          </div>
          <div className="col-span-12 md:col-span-3 font-body text-sm leading-snug">
            Same model, same calibration pipeline. The only new work per vertical is curating threat labels.
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 md:gap-4">
          {VERTICALS.map((v, i) => (
            <Card key={v.n} {...v} delay={i * 0.08} />
          ))}
        </div>
      </div>
    </section>
  );
}

function Card({
  n,
  name,
  buyer,
  unsafe,
  color,
  delay,
}: {
  n: string;
  name: string;
  buyer: string;
  unsafe: string[];
  color: string;
  delay: number;
}) {
  const [hover, setHover] = useState(false);
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ delay, duration: 0.5 }}
      onHoverStart={() => setHover(true)}
      onHoverEnd={() => setHover(false)}
      className={`relative border-3 border-ink p-5 min-h-[260px] ${color} cursor-pointer overflow-hidden`}
    >
      <motion.div
        animate={{ opacity: hover ? 0 : 1, y: hover ? -10 : 0 }}
        transition={{ duration: 0.3 }}
        className="absolute inset-0 p-5 flex flex-col justify-between"
      >
        <div className="font-display text-5xl tracking-tight leading-none">{n}</div>
        <div>
          <div className="font-display uppercase text-xl leading-tight">{name}</div>
          <div className="font-mono text-[11px] uppercase tracking-widest opacity-70 mt-1">{buyer}</div>
        </div>
      </motion.div>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: hover ? 1 : 0, y: hover ? 0 : 10 }}
        transition={{ duration: 0.3 }}
        className="absolute inset-0 p-5 flex flex-col justify-between"
      >
        <div className="font-mono text-[11px] uppercase tracking-widest">Unsafe sounds</div>
        <ul className="space-y-1 font-body text-sm">
          {unsafe.map((u) => (
            <li key={u} className="flex items-start gap-2">
              <span className="inline-block mt-2 h-1.5 w-1.5 bg-current opacity-70 flex-shrink-0" />
              <span>{u}</span>
            </li>
          ))}
        </ul>
      </motion.div>
    </motion.div>
  );
}
