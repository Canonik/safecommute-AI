// Builds web/public/demo/safecommute-v2-demo.zip — a PyTorch-free inference
// bundle: the INT8 ONNX model + a standalone Python runner + feature stats.
// Runs at `npm run prebuild`. Idempotent — rebuilds only if inputs are newer.
//
// Contents (per VALIDATE_AND_IMPROVE.md §5.4):
//   safecommute_v2_int8.onnx   static INT8 model, ~3.7 MB, single file
//   infer.py                   standalone runner (librosa + onnxruntime)
//   feature_stats.json         mean/std normalization stats
//   short.md                   architecture + privacy + fine-tune notes
//   README-bundle.md           install + run instructions

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import archiver from "archiver";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const repoRoot = path.resolve(webDir, "..");

const MODEL_INT8 = path.join(repoRoot, "models", "safecommute_v2_int8.onnx");
const MODEL_FP32 = path.join(repoRoot, "models", "safecommute_v2.onnx");
const INFER_SRC = path.join(webDir, "public", "demo", "infer.py");
const STATS_SRC = path.join(repoRoot, "feature_stats.json");
const SHORT_MD = path.join(webDir, "public", "demo", "short.md");
const README_SRC = path.join(webDir, "public", "demo", "README-bundle.md");
const OUT_DIR = path.join(webDir, "public", "demo");
const OUT_ZIP = path.join(OUT_DIR, "safecommute-v2-demo.zip");

// Prefer the INT8 ONNX. Fall back to FP32 ONNX if INT8 not present.
let MODEL_SRC = MODEL_INT8;
let MODEL_NAME = "safecommute_v2_int8.onnx";
if (!fs.existsSync(MODEL_INT8)) {
  if (fs.existsSync(MODEL_FP32)) {
    console.warn(
      `[build-demo-bundle] INT8 ONNX missing — falling back to FP32 ` +
        `${path.basename(MODEL_FP32)}. Run ` +
        `\`python -m safecommute.export_quantized\` to ship the smaller/faster INT8.`
    );
    MODEL_SRC = MODEL_FP32;
    MODEL_NAME = "safecommute_v2.onnx";
  } else {
    console.warn(
      `[build-demo-bundle] No ONNX model at ${MODEL_INT8} or ${MODEL_FP32} — ` +
        `skipping zip. Run \`python -m safecommute.export\` first.`
    );
    process.exit(0);
  }
}

for (const required of [INFER_SRC, SHORT_MD, README_SRC, STATS_SRC]) {
  if (!fs.existsSync(required)) {
    console.error(`[build-demo-bundle] missing required file: ${required}`);
    process.exit(1);
  }
}

fs.mkdirSync(OUT_DIR, { recursive: true });

const inputsMtime = Math.max(
  fs.statSync(MODEL_SRC).mtimeMs,
  fs.statSync(INFER_SRC).mtimeMs,
  fs.statSync(SHORT_MD).mtimeMs,
  fs.statSync(README_SRC).mtimeMs,
  fs.statSync(STATS_SRC).mtimeMs,
  fs.statSync(fileURLToPath(import.meta.url)).mtimeMs
);
if (fs.existsSync(OUT_ZIP) && fs.statSync(OUT_ZIP).mtimeMs >= inputsMtime) {
  console.log(`[build-demo-bundle] ${path.relative(webDir, OUT_ZIP)} up to date`);
  process.exit(0);
}

const output = fs.createWriteStream(OUT_ZIP);
const archive = archiver("zip", { zlib: { level: 9 } });

output.on("close", () => {
  const mb = (archive.pointer() / 1_048_576).toFixed(2);
  console.log(`[build-demo-bundle] wrote ${path.relative(webDir, OUT_ZIP)} (${mb} MB)`);
});
archive.on("warning", (err) => {
  if (err.code === "ENOENT") console.warn(err);
  else throw err;
});
archive.on("error", (err) => {
  throw err;
});

archive.pipe(output);
archive.file(MODEL_SRC, { name: MODEL_NAME });
archive.file(INFER_SRC, { name: "infer.py" });
archive.file(STATS_SRC, { name: "feature_stats.json" });
archive.file(SHORT_MD, { name: "short.md" });
archive.file(README_SRC, { name: "README-bundle.md" });
await archive.finalize();
