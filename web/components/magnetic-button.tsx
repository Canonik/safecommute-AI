"use client";

import { motion, useMotionValue, useSpring } from "motion/react";
import { useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type Props = {
  href?: string;
  onClick?: () => void;
  children: ReactNode;
  variant?: "primary" | "outline" | "ghost";
  className?: string;
  external?: boolean;
};

export function MagneticButton({
  href,
  onClick,
  children,
  variant = "primary",
  className,
  external = false,
}: Props) {
  const ref = useRef<HTMLAnchorElement | HTMLButtonElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 200, damping: 15 });
  const sy = useSpring(y, { stiffness: 200, damping: 15 });

  const handleMove = (e: React.MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const relX = e.clientX - (rect.left + rect.width / 2);
    const relY = e.clientY - (rect.top + rect.height / 2);
    x.set(relX * 0.25);
    y.set(relY * 0.4);
  };
  const handleLeave = () => {
    x.set(0);
    y.set(0);
  };

  const base =
    "relative inline-flex items-center justify-center px-6 py-4 font-display uppercase tracking-tight text-sm md:text-base border-3 border-ink transition-colors";
  const styles = {
    primary: "bg-ink text-paper hover:bg-bauhaus-red",
    outline: "bg-transparent text-ink hover:bg-ink hover:text-paper",
    ghost: "bg-paper text-ink hover:bg-bauhaus-yellow",
  }[variant];

  const content = (
    <motion.span
      style={{ x: sx, y: sy }}
      className="inline-flex items-center gap-2"
    >
      {children}
    </motion.span>
  );

  if (href) {
    return (
      <motion.a
        ref={ref as React.RefObject<HTMLAnchorElement>}
        href={href}
        target={external ? "_blank" : undefined}
        rel={external ? "noopener noreferrer" : undefined}
        onMouseMove={handleMove}
        onMouseLeave={handleLeave}
        whileTap={{ scale: 0.96 }}
        className={cn(base, styles, className)}
      >
        {content}
      </motion.a>
    );
  }
  return (
    <motion.button
      ref={ref as React.RefObject<HTMLButtonElement>}
      onClick={onClick}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      whileTap={{ scale: 0.96 }}
      className={cn(base, styles, className)}
    >
      {content}
    </motion.button>
  );
}
