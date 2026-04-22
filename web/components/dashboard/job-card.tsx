"use client";

import { motion } from "motion/react";
import { WaveformCanvas } from "@/components/waveform-canvas";
import { Circle } from "@/components/ui/shapes";

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

const STATUS_META: Record<JobStatus, { label: string; stripe: string; fg: string; note: string }> = {
  queued: {
    label: "Queued",
    stripe: "bg-bauhaus-yellow",
    fg: "text-ink",
    note: "Waiting for worker",
  },
  running: {
    label: "Running",
    stripe: "bg-bauhaus-blue",
    fg: "text-paper",
    note: "Fine-tuning on your clips",
  },
  succeeded: {
    label: "Ready",
    stripe: "bg-ink",
    fg: "text-paper",
    note: "Download your calibrated model",
  },
  failed: {
    label: "Failed",
    stripe: "bg-bauhaus-red",
    fg: "text-paper",
    note: "Retry or contact support",
  },
};

type Props = {
  id: string;
  status: JobStatus;
  siteName: string;
  createdAt: string;
  modelPath?: string | null;
  error?: string | null;
};

export function JobCard({ id, status, siteName, createdAt, modelPath, error }: Props) {
  const meta = STATUS_META[status];
  const isActive = status === "queued" || status === "running";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ type: "spring", stiffness: 140, damping: 18 }}
      className="relative border-3 border-ink bg-paper shadow-[6px_6px_0_0_#0a0a0a]"
    >
      <div className={`relative h-10 border-b-3 border-ink ${meta.stripe} ${meta.fg} flex items-center px-4 font-mono text-[11px] uppercase tracking-widest`}>
        {isActive && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            className="absolute -right-4 -top-4"
          >
            <Circle className="w-16 h-16 opacity-30" fill="currentColor" />
          </motion.div>
        )}
        <span className="font-display text-sm tracking-tight">{meta.label}</span>
        <span className="ml-auto opacity-70">Job · {id.slice(0, 8)}</span>
      </div>
      <div className="p-5 grid grid-cols-12 gap-4">
        <div className="col-span-12 md:col-span-7">
          <div className="font-mono text-[10px] uppercase tracking-widest opacity-60">Site</div>
          <div className="font-display text-2xl md:text-3xl leading-none tracking-tight mt-1">{siteName}</div>
          <div className="font-mono text-[11px] uppercase tracking-widest opacity-60 mt-4">Queued</div>
          <div className="font-grotesk text-sm mt-1">{new Date(createdAt).toLocaleString()}</div>
          <div className="font-grotesk text-sm italic mt-3">{meta.note}</div>
          {error && (
            <div className="mt-3 border-3 border-bauhaus-red px-3 py-2 bg-bauhaus-red/10 font-mono text-[11px]">
              {error}
            </div>
          )}
          {status === "succeeded" && modelPath && (
            <div className="mt-4 flex flex-wrap gap-2">
              <a
                href={`/api/finetune/${id}/download?file=model`}
                className="inline-flex items-center border-3 border-ink bg-ink text-paper px-4 py-2 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red transition-colors"
              >
                Model .onnx ↓
              </a>
              <a
                href={`/api/finetune/${id}/download?file=thresholds`}
                className="inline-flex items-center border-3 border-ink bg-paper text-ink px-4 py-2 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-yellow transition-colors"
              >
                Thresholds ↓
              </a>
              <a
                href={`/api/finetune/${id}/download?file=report`}
                className="inline-flex items-center border-3 border-ink bg-paper text-ink px-4 py-2 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-blue hover:text-paper transition-colors"
              >
                Deployment report ↓
              </a>
            </div>
          )}
        </div>
        <div className="col-span-12 md:col-span-5 relative border-3 border-ink h-24 md:h-full min-h-[120px]">
          <WaveformCanvas
            className="absolute inset-0 w-full h-full"
            calmColor="#0a0a0a"
            alertColor={status === "failed" ? "#e63946" : "#1d3bc1"}
          />
        </div>
      </div>
    </motion.div>
  );
}
