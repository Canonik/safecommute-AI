// Builds web/public/demo/safecommute-v2-demo.zip containing the .pth + short.md.
// Runs at `npm run prebuild`. Idempotent — rebuilds only if inputs are newer.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import archiver from "archiver";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const repoRoot = path.resolve(webDir, "..");

const MODEL_SRC = path.join(repoRoot, "models", "safecommute_v2.pth");
const SHORT_MD = path.join(webDir, "public", "demo", "short.md");
const OUT_DIR = path.join(webDir, "public", "demo");
const OUT_ZIP = path.join(OUT_DIR, "safecommute-v2-demo.zip");

if (!fs.existsSync(MODEL_SRC)) {
  console.warn(
    `[build-demo-bundle] model not found at ${MODEL_SRC} — skipping zip. ` +
      `If the pre-built zip is already checked in, that one will ship. ` +
      `Otherwise: train/copy the model before deploy.`
  );
  process.exit(0);
}
if (!fs.existsSync(SHORT_MD)) {
  console.error(`[build-demo-bundle] missing ${SHORT_MD}`);
  process.exit(1);
}

fs.mkdirSync(OUT_DIR, { recursive: true });

const inputsMtime = Math.max(
  fs.statSync(MODEL_SRC).mtimeMs,
  fs.statSync(SHORT_MD).mtimeMs,
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
archive.file(MODEL_SRC, { name: "safecommute_v2.pth" });
archive.file(SHORT_MD, { name: "short.md" });
await archive.finalize();
