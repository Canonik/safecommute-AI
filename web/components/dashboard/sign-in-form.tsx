"use client";

import { motion } from "motion/react";
import { useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/client";
import { Circle, Square, Triangle } from "@/components/ui/shapes";

export function SignInForm({ nextPath }: { nextPath: string }) {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("sending");
    setError(null);
    try {
      const supabase = supabaseBrowser();
      const redirectTo = `${window.location.origin}/api/auth/callback?next=${encodeURIComponent(nextPath)}`;
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: redirectTo },
      });
      if (error) throw error;
      setStatus("sent");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  };

  return (
    <div className="relative border-6 border-ink bg-paper p-8 md:p-12 shadow-[12px_12px_0_0_#0a0a0a]">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
        className="absolute -top-8 -left-8 pointer-events-none"
      >
        <Circle className="w-32 h-32" fill="#ffd100" />
      </motion.div>
      <div className="absolute -bottom-6 -right-6 pointer-events-none">
        <Square className="w-24 h-24" fill="#1d3bc1" />
      </div>
      <div className="relative z-10">
        <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest mb-4">
          <Triangle className="w-3 h-3" fill="#e63946" />
          <span>Sign in / up</span>
        </div>
        <motion.h1
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ type: "spring", stiffness: 140, damping: 18 }}
          className="font-display uppercase text-5xl md:text-6xl leading-[0.9] tracking-tight"
        >
          Open the <br />
          <span className="text-bauhaus-red">dashboard</span>
        </motion.h1>
        <p className="font-grotesk text-sm mt-5 max-w-md">
          We send a magic link to your email. No passwords. New users get an account automatically.
        </p>

        {status === "sent" ? (
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 180, damping: 16 }}
            className="mt-8 border-3 border-ink bg-bauhaus-yellow p-6"
          >
            <div className="font-display uppercase text-2xl">Check your email</div>
            <div className="font-grotesk text-sm mt-2">
              Sent a magic link to <strong>{email}</strong>. Click it to land on your dashboard.
            </div>
          </motion.div>
        ) : (
          <form onSubmit={submit} className="mt-8">
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full border-3 border-ink bg-paper px-4 py-4 font-grotesk text-base focus:outline-none focus:bg-bauhaus-yellow transition-colors"
            />
            <button
              type="submit"
              disabled={status === "sending" || !email}
              className="mt-4 w-full border-3 border-ink bg-ink text-paper px-5 py-4 font-display uppercase text-base tracking-tight hover:bg-bauhaus-red disabled:opacity-50 transition-colors"
            >
              {status === "sending" ? "Sending…" : "Send magic link →"}
            </button>
            {error && (
              <div className="mt-3 border-3 border-bauhaus-red bg-bauhaus-red/10 px-3 py-2 font-mono text-[11px]">
                {error}
              </div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
