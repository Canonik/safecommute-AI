import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#f4f1ea",
        ink: "#0a0a0a",
        bauhaus: {
          red: "#e63946",
          "red-deep": "#c1121f",
          blue: "#1d3bc1",
          "blue-deep": "#0b2a9c",
          yellow: "#ffd100",
          "yellow-warm": "#f9c74f",
        },
      },
      fontFamily: {
        display: ["var(--font-archivo-black)", "system-ui", "sans-serif"],
        grotesk: ["var(--font-space-grotesk)", "system-ui", "sans-serif"],
        bricolage: ["var(--font-bricolage)", "system-ui", "sans-serif"],
        body: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "mega": ["clamp(3rem, 10vw, 10rem)", { lineHeight: "0.9", letterSpacing: "-0.04em" }],
        "giga": ["clamp(4rem, 14vw, 14rem)", { lineHeight: "0.85", letterSpacing: "-0.05em" }],
      },
      borderWidth: {
        "3": "3px",
        "6": "6px",
      },
    },
  },
  plugins: [],
};

export default config;
