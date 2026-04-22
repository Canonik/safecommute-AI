"""Per-job processor for the SafeCommute fine-tune worker.

Given a claimed `finetune_jobs` row (status already flipped to 'running'),
this module:

  1. Downloads the site's uploaded clips from the `audio-uploads` bucket into
     a per-job temp dir, split 80/20 into (training ambient | calibration).
  2. Runs `safecommute/pipeline/finetune.py` with the tweak-3 recipe:
       --keep-safe-ratio 0.1 --epochs 20 --freeze-cnn
       --calibration-ambient-dir <tmp>/calibration/
     This is the best architecture-preserving recipe found in Phase 1
     (paper.md §1.4 tweak 3). The --calibration-ambient-dir flag produces a
     `low_fpr_site` threshold so the deployed model's gate is calibrated on
     the customer's own ambient, not the universal test set.
  3. Runs `safecommute/export_quantized.py` to produce the INT8 ONNX artefact
     (~3.7 MB, matches paper §3.3 SOTA row).
  4. Runs `safecommute/pipeline/test_deployment.py` on the customer ambient
     at `--majority-k 2` for a deployment report JSON.
  5. Uploads model.onnx + thresholds.json + deployment_report.json to
     `models-deliverable/{owner}/{job_id}/` (private bucket, RLS'd).
  6. Updates the job row: status='succeeded', model_path=<storage prefix>,
     thresholds=<jsonb of the full thresholds file>, completed_at=now().
  7. Deletes the raw clips from `audio-uploads` — the **privacy fix**
     described in paper.md §5 / privacy-section.tsx: raw audio is ephemeral;
     only the non-invertible PCEN-derived model weights persist.

Anything that throws becomes a status='failed' row with a short error message
and the full traceback in `worker_logs`. The customer sees the message; the
operator sees the trace.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Tuple

from .env import env_float, env_int, env_str
from .export import export_int8_onnx
from .supabase_client import Supabase

log = logging.getLogger(__name__)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CLIPS_BUCKET = env_str("CLIPS_BUCKET", "audio-uploads")
MODEL_BUCKET = env_str("MODEL_BUCKET", "models-deliverable")

# Tweak-3 recipe (paper.md §1.4). Kept as constants so the worker's recipe is
# visible at a glance and overridable per env (e.g. for cheaper dev runs).
# env_{int,float,str} strip any accidental inline `# comment` from a
# systemd EnvironmentFile line — see worker/env.py for why that matters.
FT_KEEP_SAFE_RATIO = env_float("FT_KEEP_SAFE_RATIO", 0.1)
FT_EPOCHS = env_int("FT_EPOCHS", 20)
FT_LR = env_float("FT_LR", 1e-4)
MAJORITY_K = env_int("MAJORITY_K", 2)


def _split_clips_80_20(clips: List[Dict[str, Any]], salt: str
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic 80/20 basename-hash split — matches
    tests/validate_fp_claims.sha256_bucket_80_20. The 20% held-out portion is
    used ONLY to calibrate the `low_fpr_site` threshold in finetune.py, never
    for training.
    """
    training, calib = [], []
    for c in clips:
        h = int(hashlib.sha256((salt + c["filename"]).encode()).hexdigest(),
                16) % 100
        (calib if h >= 80 else training).append(c)
    # Guarantee at least one wav in each bucket so Step 7b in finetune.py
    # has something to calibrate against, even with the minimum 3-clip
    # requirement from the trigger route.
    if not calib and len(training) >= 2:
        calib.append(training.pop())
    if not training and calib:
        training.append(calib.pop())
    return training, calib


def _download_clip(sb: Supabase, clip: Dict[str, Any], dest_dir: str) -> str:
    """Download one clip from audio-uploads. Returns the local path."""
    safe_name = clip["filename"]
    local = os.path.join(dest_dir, safe_name)
    sb.storage_download(CLIPS_BUCKET, clip["storage_path"], local)
    return local


def _run_pipeline(env_name: str, ambient_dir: str, calib_dir: str,
                  save_model: str, save_thresh: str) -> Dict[str, Any]:
    """Shell out to safecommute/pipeline/finetune.py with the tweak-3 recipe.

    Writes `save_model` (.pth) and `models/{env_name}_thresholds.json` which
    we rename to `save_thresh` afterwards. Returns {"stdout", "returncode"}.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    cmd = [
        sys.executable, "-u",
        os.path.join(REPO, "safecommute", "pipeline", "finetune.py"),
        "--environment", env_name,
        "--ambient-dir", ambient_dir,
        "--freeze-cnn",
        "--keep-safe-ratio", str(FT_KEEP_SAFE_RATIO),
        "--epochs", str(FT_EPOCHS),
        "--lr", str(FT_LR),
    ]
    if os.path.isdir(calib_dir) and os.listdir(calib_dir):
        cmd += ["--calibration-ambient-dir", calib_dir,
                "--calibration-majority-k", str(MAJORITY_K)]

    log.info("running %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=3600)
    # Always log the subprocess output so failures are diagnosable from the
    # worker's journal, not just the raised exception.
    if result.stdout:
        log.info("finetune.py stdout tail:\n%s", result.stdout[-2000:])
    if result.stderr:
        log.info("finetune.py stderr tail:\n%s", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(
            f"finetune.py exit {result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}")

    # finetune.py always writes to models/{env_name}_model.pth — move it to
    # the target path our caller asked for (keeps the worker's per-job
    # artefacts out of the model registry).
    produced_pth = os.path.join(REPO, "models", f"{env_name}_model.pth")
    produced_thresh = os.path.join(REPO, "models", f"{env_name}_thresholds.json")
    if not os.path.exists(produced_pth):
        raise RuntimeError(
            f"finetune.py exit 0 but {produced_pth} missing — this usually "
            f"means 0 ambient chunks were processed (unsupported file "
            f"extension, unreadable audio, or all clips < 1s). stdout tail:"
            f"\n{result.stdout[-2000:]}")
    shutil.move(produced_pth, save_model)
    shutil.move(produced_thresh, save_thresh)

    return {"stdout_tail": result.stdout[-2000:]}


def _export_int8_onnx(pth_path: str, onnx_out: str) -> None:
    """Produce the INT8 ONNX artefact for delivery. Uses worker.export (thin
    wrapper around safecommute.export_quantized helpers) so per-job output
    paths don't collide with the base-model INT8 ONNX."""
    log.info("exporting INT8 ONNX: %s -> %s", pth_path, onnx_out)
    stats = export_int8_onnx(pth_path, onnx_out)
    log.info("INT8 ONNX size=%.2f MB logit |Δ| max=%.4f (%d samples checked)",
             stats["int8_mb"], stats["logit_max_diff"], stats["n_checked"])


def _deployment_report(pth_path: str, thresh_path: str,
                       ambient_dir: str,
                       threat_dir: str) -> Dict[str, Any]:
    """Run test_deployment.py and capture the passed/failed-per-test summary.

    Uses majority-k=2 + the per-site threshold produced by finetune.py. The
    report is stashed alongside the model artefact so paying customers can
    see which gates their fine-tune met (and which didn't, honestly).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    cmd = [
        sys.executable, "-u",
        os.path.join(REPO, "safecommute", "pipeline", "test_deployment.py"),
        "--model", pth_path,
        "--thresholds-file", thresh_path,
        "--threat-dir", threat_dir,
        "--ambient-dir", ambient_dir,
        "--majority-k", str(MAJORITY_K),
        "--verbose",
    ]
    log.info("running %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=600)
    # Non-zero exit means one or more must-pass gates failed. That is NOT a
    # worker failure — the customer should see the report and decide. We
    # capture the stdout and echo it back as the report.
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr_tail": result.stderr[-1000:],
    }


def process_job(sb: Supabase, job: Dict[str, Any]) -> None:
    """Run a single claimed job end-to-end. Raises on fatal worker errors
    (those bubble up to main.py's try/except which marks the job failed)."""
    job_id = job["id"]
    site_id = job["site_id"]
    owner = job["owner"]
    env_name = f"job-{job_id}"  # fed to finetune.py --environment

    log.info("processing job %s (site=%s owner=%s)", job_id, site_id, owner)

    clips = sb.list_clips_for_site(site_id)
    if len(clips) < 3:
        raise RuntimeError(f"need ≥3 clips, have {len(clips)}")

    with tempfile.TemporaryDirectory(prefix=f"safecommute_job_{job_id}_") as tmp:
        train_dir = os.path.join(tmp, "ambient_fit")
        cal_dir = os.path.join(tmp, "ambient_cal")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(cal_dir, exist_ok=True)

        # ── 1. Deterministic 80/20 split (same hashing strategy as tests/) ──
        salt = f"job:{job_id}"
        training, calib = _split_clips_80_20(clips, salt)
        log.info("split: %d training / %d calibration", len(training), len(calib))

        for c in training:
            _download_clip(sb, c, train_dir)
        for c in calib:
            _download_clip(sb, c, cal_dir)

        # ── 2. Fine-tune ──
        artefacts_dir = os.path.join(tmp, "artefacts")
        os.makedirs(artefacts_dir, exist_ok=True)
        pth_path = os.path.join(artefacts_dir, "model.pth")
        thresh_path = os.path.join(artefacts_dir, "thresholds.json")
        _run_pipeline(env_name, train_dir, cal_dir, pth_path, thresh_path)

        # ── 3. INT8 ONNX export (the real deployment artefact) ──
        onnx_path = os.path.join(artefacts_dir, "model.onnx")
        _export_int8_onnx(pth_path, onnx_path)

        # ── 4. Deployment report ──
        # We use the customer's own calibration clips as the "ambient" for
        # the deployment gate — that's the honest per-site measurement.
        report = _deployment_report(pth_path, thresh_path,
                                    ambient_dir=cal_dir,
                                    threat_dir=os.path.join(
                                        REPO, "raw_data", "youtube_screams"))
        report_path = os.path.join(artefacts_dir, "deployment_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # ── 5. Upload to models-deliverable bucket ──
        prefix = f"{owner}/{job_id}"
        sb.storage_upload(MODEL_BUCKET, f"{prefix}/model.onnx",
                          onnx_path, content_type="application/octet-stream")
        sb.storage_upload(MODEL_BUCKET, f"{prefix}/thresholds.json",
                          thresh_path, content_type="application/json")
        sb.storage_upload(MODEL_BUCKET, f"{prefix}/deployment_report.json",
                          report_path, content_type="application/json")
        log.info("uploaded artefacts to %s/%s/", MODEL_BUCKET, prefix)

        # ── 6. Mark job succeeded ──
        with open(thresh_path) as f:
            thresholds_json = json.load(f)
        sb.mark_job_succeeded(job_id,
                              model_path=prefix,
                              thresholds=thresholds_json)

        # ── 7. Privacy fix (paper §5 / privacy-section.tsx): delete raw
        # audio now that the model has been produced. PCEN is applied
        # inside finetune.py so nothing downstream needs the wavs, and
        # leaving them in Supabase contradicts "audio does not persist".
        clip_paths = [c["storage_path"] for c in clips]
        sb.storage_delete(CLIPS_BUCKET, clip_paths)
        log.info("deleted %d source clips from %s", len(clip_paths),
                 CLIPS_BUCKET)


def run_job_safely(sb: Supabase, job: Dict[str, Any]) -> None:
    """Wrap process_job. On uncaught exceptions, mark the job failed and
    persist the traceback in worker_logs."""
    job_id = job["id"]
    try:
        process_job(sb, job)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log.exception("job %s failed: %s", job_id, e)
        short = f"{type(e).__name__}: {e}"
        try:
            sb.insert_worker_log(job_id, level="error", message=short,
                                 traceback=tb)
            sb.mark_job_failed(job_id, error=short)
        except Exception:  # noqa: BLE001
            log.exception("also failed to mark job failed — manual cleanup needed")
