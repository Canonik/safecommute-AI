"use client";

import { useEffect, useRef } from "react";

type Props = {
  className?: string;
  calmColor?: string;
  alertColor?: string;
  bg?: string;
};

export function WaveformCanvas({
  className,
  calmColor = "#0a0a0a",
  alertColor = "#e63946",
  bg = "transparent",
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let rafId = 0;
    let start = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = (now: number) => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const t = (now - start) / 1000;

      ctx.clearRect(0, 0, w, h);
      if (bg !== "transparent") {
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);
      }

      // escalation spike window: every 8s, 1.2s duration
      const cycle = t % 8;
      const inSpike = cycle > 4 && cycle < 5.2;
      const spikeT = inSpike ? (cycle - 4) / 1.2 : 0; // 0..1
      const spikeEnv = inSpike
        ? Math.sin(spikeT * Math.PI) // 0..1..0
        : 0;

      // draw centered oscilloscope
      const mid = h / 2;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.strokeStyle = inSpike ? alertColor : calmColor;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      const step = 2;
      for (let x = 0; x <= w; x += step) {
        const p = x / w;
        // base: gentle sine + low-amp noise
        const base =
          Math.sin(p * 14 + t * 1.4) * 6 +
          Math.sin(p * 37 + t * 2.1) * 3 +
          (Math.random() - 0.5) * 2;
        // spike: heavy amplitude + chaotic transients
        const spike =
          spikeEnv *
          (Math.sin(p * 80 + t * 30) * 60 +
            (Math.random() - 0.5) * 50 * spikeEnv);
        const y = mid + base + spike;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // under-trace fill (faint) on spike only
      if (inSpike) {
        ctx.globalAlpha = 0.08;
        ctx.fillStyle = alertColor;
        ctx.fillRect(0, 0, w, h);
        ctx.globalAlpha = 1;
      }

      if (!reduced) rafId = requestAnimationFrame(draw);
    };

    if (reduced) {
      // single static frame
      draw(performance.now());
    } else {
      rafId = requestAnimationFrame(draw);
    }

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, [calmColor, alertColor, bg]);

  return <canvas ref={ref} className={className} aria-hidden />;
}
