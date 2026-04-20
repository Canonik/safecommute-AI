"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, useMotionValue, useSpring } from "motion/react";
import { useEffect, useRef, type ReactNode } from "react";
import { Circle, Square, Triangle } from "@/components/ui/shapes";
import { supabaseBrowser } from "@/lib/supabase/client";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Overview", Shape: Circle },
  { href: "/dashboard/sites", label: "Sites", Shape: Square },
  { href: "/dashboard/billing", label: "Billing", Shape: Triangle },
] as const;

export function DashboardShell({
  userEmail,
  children,
}: {
  userEmail: string | null;
  children: ReactNode;
}) {
  const pathname = usePathname();
  const containerRef = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 70, damping: 14 });
  const sy = useSpring(my, { stiffness: 70, damping: 14 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      mx.set(e.clientX - r.left);
      my.set(e.clientY - r.top);
    };
    el.addEventListener("mousemove", onMove);
    return () => el.removeEventListener("mousemove", onMove);
  }, [mx, my]);

  const signOut = async () => {
    const supabase = supabaseBrowser();
    await supabase.auth.signOut();
    window.location.href = "/";
  };

  return (
    <div ref={containerRef} className="relative min-h-screen bg-paper text-ink overflow-hidden">
      <motion.div
        style={{ x: sx, y: sy, translateX: "-50%", translateY: "-50%" }}
        className="pointer-events-none fixed z-0 w-[420px] h-[420px] rounded-full bg-bauhaus-yellow opacity-60 mix-blend-multiply blur-[1px] hidden md:block"
      />
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 80, repeat: Infinity, ease: "linear" }}
        className="pointer-events-none fixed -right-48 -bottom-48 z-0 hidden md:block"
      >
        <Circle className="w-[600px] h-[600px]" fill="#1d3bc1" />
      </motion.div>

      {/* top bar */}
      <header className="relative z-20 border-b-6 border-ink bg-paper">
        <div className="mx-auto max-w-[1400px] px-4 md:px-8 h-16 flex items-center gap-4">
          <Link href="/" className="flex items-center gap-3 group">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
            >
              <Circle className="w-6 h-6" fill="#e63946" />
            </motion.div>
            <span className="font-display uppercase text-lg tracking-tight">SafeCommute</span>
            <span className="font-mono text-[10px] uppercase tracking-widest bg-ink text-paper px-2 py-0.5">
              dashboard
            </span>
          </Link>
          <nav className="ml-6 hidden md:flex items-center gap-1">
            {NAV.map(({ href, label, Shape }) => {
              const active =
                href === "/dashboard" ? pathname === href : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "relative flex items-center gap-2 px-3 py-2 border-3 border-transparent font-display uppercase text-xs tracking-tight transition-colors",
                    active
                      ? "border-ink bg-bauhaus-yellow"
                      : "hover:border-ink hover:bg-paper"
                  )}
                >
                  <Shape className="w-3 h-3" fill="#0a0a0a" />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {userEmail && (
              <span className="hidden md:inline font-mono text-[11px] uppercase tracking-widest opacity-70">
                {userEmail}
              </span>
            )}
            <button
              onClick={signOut}
              className="border-3 border-ink px-3 py-1.5 font-display uppercase text-xs tracking-tight hover:bg-ink hover:text-paper transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10">
        <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-10 md:py-16">{children}</div>
      </main>

      <footer className="relative z-10 border-t-6 border-ink bg-ink text-paper">
        <div className="mx-auto max-w-[1400px] px-4 md:px-8 py-6 flex flex-wrap items-center gap-4 font-mono text-[11px] uppercase tracking-widest">
          <span>SafeCommute · Dashboard</span>
          <span className="opacity-50">·</span>
          <Link href="/" className="hover:text-bauhaus-yellow">
            Back to site
          </Link>
        </div>
      </footer>
    </div>
  );
}
