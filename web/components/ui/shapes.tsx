"use client";

import { motion, type MotionProps } from "motion/react";
import { cn } from "@/lib/utils";

type ShapeProps = {
  className?: string;
  fill?: string;
} & MotionProps;

export function Circle({ className, fill = "var(--bauhaus-yellow)", ...rest }: ShapeProps) {
  return (
    <motion.svg
      viewBox="0 0 100 100"
      className={cn("block", className)}
      aria-hidden
      {...rest}
    >
      <circle cx="50" cy="50" r="50" fill={fill} />
    </motion.svg>
  );
}

export function Square({ className, fill = "var(--bauhaus-red)", ...rest }: ShapeProps) {
  return (
    <motion.svg
      viewBox="0 0 100 100"
      className={cn("block", className)}
      aria-hidden
      {...rest}
    >
      <rect width="100" height="100" fill={fill} />
    </motion.svg>
  );
}

export function Triangle({ className, fill = "var(--bauhaus-blue)", ...rest }: ShapeProps) {
  return (
    <motion.svg
      viewBox="0 0 100 100"
      className={cn("block", className)}
      aria-hidden
      {...rest}
    >
      <polygon points="50,5 95,95 5,95" fill={fill} />
    </motion.svg>
  );
}
