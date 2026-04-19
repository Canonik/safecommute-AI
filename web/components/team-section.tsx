"use client";

import { motion } from "motion/react";

const MEMBERS = [
  {
    n: "01",
    name: "Alessandro Canonico",
    role: "Project Lead & AI Strategist",
    shape: "circle" as const,
    color: "#ffd100",
  },
  {
    n: "02",
    name: "Fabiola Martignetti",
    role: "Behavioral Data & ML Specialist",
    shape: "square" as const,
    color: "#e63946",
  },
  {
    n: "03",
    name: "Robbie Urquhart",
    role: "Machine Learning & Edge Engineer",
    shape: "triangle" as const,
    color: "#1d3bc1",
  },
];

export function TeamSection() {
  return (
    <section id="team" className="relative border-b-6 border-ink bg-paper">
      <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-16 md:py-24">
        <div className="grid grid-cols-12 gap-6 items-end mb-10">
          <div className="col-span-12 md:col-span-9">
            <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
              <span className="inline-block h-3 w-3 bg-ink" />
              <span>Team · Bocconi University</span>
            </div>
            <h2 className="font-display uppercase text-5xl md:text-7xl leading-[0.9] tracking-tight">
              Three people. <br />
              <span className="text-bauhaus-red">One deployable model.</span>
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          {MEMBERS.map((m, i) => (
            <motion.div
              key={m.n}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: i * 0.1, duration: 0.6 }}
              className="border-3 border-ink bg-paper p-6 relative overflow-hidden group"
            >
              <div className="flex items-start justify-between mb-6">
                <div className="font-display text-5xl leading-none">{m.n}</div>
                <div className="w-16 h-16 relative">
                  {m.shape === "circle" && (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
                      className="absolute inset-0 rounded-full"
                      style={{ background: m.color }}
                    />
                  )}
                  {m.shape === "square" && (
                    <motion.div
                      animate={{ rotate: [0, 90, 90, 0] }}
                      transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                      className="absolute inset-0"
                      style={{ background: m.color }}
                    />
                  )}
                  {m.shape === "triangle" && (
                    <div
                      className="absolute inset-0"
                      style={{ background: m.color, clipPath: "polygon(50% 0, 100% 100%, 0 100%)" }}
                    />
                  )}
                </div>
              </div>
              <div className="font-display uppercase text-2xl leading-tight mb-1">{m.name}</div>
              <div className="font-mono text-[11px] uppercase tracking-widest opacity-70">{m.role}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
