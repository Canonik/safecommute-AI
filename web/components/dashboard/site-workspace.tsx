"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { useCallback, useRef, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/client";
import { JobCard, type JobStatus } from "@/components/dashboard/job-card";
import { Circle, Square, Triangle } from "@/components/ui/shapes";
import { MagneticButton } from "@/components/magnetic-button";

type Clip = {
  id: string;
  filename: string;
  size_bytes: number | null;
  duration_s: number | null;
  uploaded_at: string;
};

type Job = {
  id: string;
  status: JobStatus;
  model_path: string | null;
  error: string | null;
  created_at: string;
};

type Site = { id: string; name: string; environment: string };

export function SiteWorkspace({
  site,
  clips: initialClips,
  jobs: initialJobs,
  hasCredit,
  subscriptionActive,
  perRunCredits,
}: {
  site: Site;
  clips: Clip[];
  jobs: Job[];
  hasCredit: boolean;
  subscriptionActive: boolean;
  perRunCredits: number;
}) {
  const [clips, setClips] = useState<Clip[]>(initialClips);
  const [jobs, setJobs] = useState<Job[]>(initialJobs);
  const [uploading, setUploading] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [triggering, setTriggering] = useState(false);
  const dragRef = useRef<HTMLLabelElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      setError(null);
      const arr = Array.from(files);
      for (const file of arr) {
        if (file.size > 25 * 1024 * 1024) {
          setError(`${file.name} is larger than 25 MB`);
          continue;
        }
        setUploading((u) => [...u, file.name]);
        try {
          const signRes = await fetch("/api/clips/upload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              site_id: site.id,
              filename: file.name,
              content_type: file.type || "audio/wav",
              size_bytes: file.size,
            }),
          });
          if (!signRes.ok) {
            const j = await signRes.json().catch(() => ({ error: "sign failed" }));
            throw new Error(j.error ?? "sign failed");
          }
          const { upload_url, path } = await signRes.json();
          const put = await fetch(upload_url, {
            method: "PUT",
            headers: { "Content-Type": file.type || "audio/wav" },
            body: file,
          });
          if (!put.ok) throw new Error(`upload failed: ${put.status}`);

          setClips((c) => [
            {
              id: path,
              filename: file.name,
              size_bytes: file.size,
              duration_s: null,
              uploaded_at: new Date().toISOString(),
            },
            ...c,
          ]);
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        } finally {
          setUploading((u) => u.filter((n) => n !== file.name));
        }
      }
    },
    [site.id]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) void handleFiles(e.dataTransfer.files);
  };

  const triggerFinetune = async () => {
    if (!hasCredit || clips.length < 3 || triggering) return;
    setTriggering(true);
    setError(null);
    try {
      const res = await fetch("/api/finetune/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ site_id: site.id }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({ error: "trigger failed" }));
        throw new Error(j.error ?? "trigger failed");
      }
      const { job_id } = await res.json();
      setJobs((j) => [
        {
          id: job_id,
          status: "queued",
          model_path: null,
          error: null,
          created_at: new Date().toISOString(),
        },
        ...j,
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Upload panel */}
      <div className="col-span-12 lg:col-span-7">
        <label
          ref={dragRef}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`relative block border-6 border-dashed border-ink p-8 md:p-14 text-center cursor-pointer overflow-hidden transition-colors ${
            dragOver ? "bg-bauhaus-yellow" : "bg-paper hover:bg-bauhaus-yellow/30"
          }`}
        >
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.m4a"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
          />
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 22, repeat: Infinity, ease: "linear" }}
            className="absolute -right-10 -top-10 pointer-events-none"
          >
            <Circle className="w-40 h-40 opacity-40" fill="#ffd100" />
          </motion.div>
          <motion.div
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute left-6 bottom-6 pointer-events-none"
          >
            <Triangle className="w-10 h-10" fill="#e63946" />
          </motion.div>
          <div className="relative z-10">
            <div className="font-mono text-[11px] uppercase tracking-widest mb-3">Drop audio</div>
            <div className="font-display uppercase text-3xl md:text-5xl leading-[0.95]">
              Upload ambient clips
            </div>
            <div className="font-grotesk text-sm mt-4 max-w-md mx-auto">
              WAV / MP3 / M4A / MP4 / MOV / FLAC · 16 kHz mono preferred · 25 MB per file · min 3
              clips recommended to kick off a tune.
            </div>
          </div>
        </label>
        {uploading.length > 0 && (
          <div className="mt-3 font-mono text-[11px] uppercase tracking-widest opacity-70">
            Uploading: {uploading.join(", ")}
          </div>
        )}
        {error && (
          <div className="mt-3 border-3 border-bauhaus-red bg-bauhaus-red/10 px-3 py-2 font-mono text-[11px]">
            {error}
          </div>
        )}

        <div className="mt-8 flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest">
          <span className="inline-block h-3 w-3 bg-bauhaus-blue" />
          <span>Clips ({clips.length})</span>
        </div>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          {clips.length === 0 && (
            <div className="md:col-span-2 border-3 border-dashed border-ink p-6 text-center font-grotesk text-sm opacity-70">
              No clips yet.
            </div>
          )}
          {clips.map((c, i) => (
            <motion.div
              key={c.id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className="border-3 border-ink p-3 flex items-center gap-3 bg-paper"
            >
              <Square className="w-4 h-4 flex-shrink-0" fill="#0a0a0a" />
              <div className="flex-1 min-w-0">
                <div className="font-display uppercase text-sm truncate">{c.filename}</div>
                <div className="font-mono text-[10px] uppercase tracking-widest opacity-70">
                  {c.size_bytes ? (c.size_bytes / 1024 / 1024).toFixed(2) + " MB" : "—"}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Run panel */}
      <div className="col-span-12 lg:col-span-5">
        <div className="relative border-3 border-ink bg-ink text-paper p-6 md:p-8 shadow-[8px_8px_0_0_#0a0a0a]">
          <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4 opacity-80">
            <Circle className="w-3 h-3" fill="#ffd100" />
            <span>Run</span>
          </div>
          <div className="font-display uppercase text-3xl md:text-4xl leading-[0.95]">
            Fine-tune this site
          </div>
          <ul className="mt-5 space-y-2 font-grotesk text-sm">
            <li className="flex gap-2">
              <span>•</span>
              <span>
                Clips uploaded: <strong>{clips.length}</strong> (≥3 required)
              </span>
            </li>
            <li className="flex gap-2">
              <span>•</span>
              <span>
                {subscriptionActive
                  ? "Subscription active — unlimited runs"
                  : perRunCredits > 0
                    ? `${perRunCredits} per-run credit${perRunCredits === 1 ? "" : "s"} available`
                    : "No credits — buy one on Billing"}
              </span>
            </li>
          </ul>
          <div className="mt-6">
            <MagneticButton
              onClick={triggerFinetune}
              variant="ghost"
              className={`w-full justify-center ${
                !hasCredit || clips.length < 3 ? "opacity-50 pointer-events-none" : ""
              }`}
            >
              {triggering ? "Queuing…" : "Run fine-tune →"}
            </MagneticButton>
          </div>
          {!hasCredit && (
            <div className="mt-4">
              <Link
                href="/dashboard/billing"
                className="block border-3 border-paper text-center px-4 py-2 font-display uppercase text-xs tracking-tight hover:bg-bauhaus-red hover:border-bauhaus-red transition-colors"
              >
                Buy credit →
              </Link>
            </div>
          )}
        </div>

        <div className="mt-6 flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest">
          <span className="inline-block h-3 w-3 bg-bauhaus-red" />
          <span>Jobs for this site</span>
        </div>
        <div className="mt-3 space-y-4">
          {jobs.length === 0 && (
            <div className="border-3 border-dashed border-ink p-6 text-center font-grotesk text-sm opacity-70">
              No runs yet.
            </div>
          )}
          {jobs.map((j) => (
            <JobCard
              key={j.id}
              id={j.id}
              status={j.status}
              siteName={site.name}
              createdAt={j.created_at}
              modelPath={j.model_path}
              error={j.error}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
