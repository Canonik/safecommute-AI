"use client";

import { motion } from "motion/react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/client";

const ENVS = ["metro", "retail", "school", "eldercare", "industrial", "other"] as const;

export function NewSiteForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [env, setEnv] = useState<(typeof ENVS)[number]>("metro");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    setError(null);
    const supabase = supabaseBrowser();
    const { data: auth } = await supabase.auth.getUser();
    const user = auth.user;
    if (!user) {
      setError("Not authenticated");
      setSaving(false);
      return;
    }
    const { data, error } = await supabase
      .from("sites")
      .insert({ owner: user.id, name: name.trim(), environment: env })
      .select("id")
      .single();
    if (error || !data) {
      setError(error?.message ?? "Could not create site");
      setSaving(false);
      return;
    }
    router.push(`/dashboard/sites/${data.id}`);
  };

  return (
    <motion.form
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      onSubmit={submit}
      className="max-w-xl border-3 border-ink bg-paper p-6 md:p-8 shadow-[6px_6px_0_0_#0a0a0a]"
    >
      <label className="block mb-6">
        <div className="font-mono text-[10px] uppercase tracking-widest mb-2">Site name</div>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Milano Centrale — platform 4"
          className="w-full border-3 border-ink bg-paper px-3 py-3 font-grotesk text-base focus:outline-none focus:bg-bauhaus-yellow transition-colors"
          required
          maxLength={120}
        />
      </label>
      <div className="mb-6">
        <div className="font-mono text-[10px] uppercase tracking-widest mb-2">Environment</div>
        <div className="grid grid-cols-3 gap-2">
          {ENVS.map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setEnv(v)}
              className={`border-3 border-ink px-3 py-2 font-display uppercase text-xs tracking-tight transition-colors ${
                env === v ? "bg-ink text-paper" : "bg-paper hover:bg-bauhaus-yellow"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>
      {error && (
        <div className="mb-4 border-3 border-bauhaus-red bg-bauhaus-red/10 px-3 py-2 font-mono text-[11px]">
          {error}
        </div>
      )}
      <button
        type="submit"
        disabled={saving || !name.trim()}
        className="border-3 border-ink bg-ink text-paper px-5 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red disabled:opacity-50 transition-colors"
      >
        {saving ? "Creating…" : "Create site →"}
      </button>
    </motion.form>
  );
}
