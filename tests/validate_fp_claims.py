"""
Validate the FP-rate claims the repo makes about safecommute_v2.

Phase A (universal): base model on `prepared_data/test/` with speech-FP ~72%.
Phase B (deployment prep): ensure a per-site fine-tuned checkpoint exists.
                           If missing and --skip-finetune not set, run
                           `safecommute/pipeline/finetune.py` on --ambient-dir.
Phase C (deployment measurement): run the deployment acceptance tests from
                           `safecommute/pipeline/test_deployment.py` against
                           the **held-out** ambient (either --held-out-ambient-dir
                           or a sha256-deterministic 80/20 split of --ambient-dir)
                           with the calibrated `low_fpr` threshold.

Phase C fix (VALIDATE_AND_IMPROVE.md §5.0): the old version measured
fine-tuned FPR on the universal `prepared_data/test/0_safe/` — the wrong
denominator. The 5% / 4.2% / 86% claims are about held-out ambient from
the target site and threat recall on real threat .wav files.

Writes a machine-readable blob to tests/reports/phase_b_<env>.json that
tests/verify_performance_claims.py reads to upgrade its Phase B rows
from WARN to PASS/FAIL.

Run (full pipeline on metro with the quarantine bucket as held-out):
    PYTHONPATH=. python tests/validate_fp_claims.py --phase c \
        --environment metro \
        --ambient-dir raw_data/youtube_metro \
        --held-out-ambient-dir raw_data/youtube_metro_quarantine \
        --threat-dir raw_data/youtube_screams

Skip fine-tuning (expect models/<env>_model.pth present):
    PYTHONPATH=. python tests/validate_fp_claims.py --skip-finetune

Only the base-model claim (no finetune needed):
    PYTHONPATH=. python tests/validate_fp_claims.py --phase a

Exit 0 = all hard claims validated, exit 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests._common import (  # noqa: E402
    all_safe_indices, load_model, load_stats, load_test_dataset,
    run_inference, safe_indices_by_prefix,
)
from safecommute.constants import MODEL_SAVE_PATH  # noqa: E402

# Claim tolerances
BASE_SPEECH_FPR_LOW = 0.60
BASE_SPEECH_FPR_HIGH = 0.85          # claim 1: ~0.72
POST_FT_FP_MAX = 0.05                # claim 2: ≤ 5% on held-out
POST_FT_SPEECH_FP_MAX = 0.10         # claim: 4.2% (wider tolerance)
POST_FT_THREAT_RECALL_MIN = 0.86     # claim: ~86%

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(REPO_ROOT, 'tests', 'reports')


# ──────────────────────────────────────────────────────────────────────
# Phase A — base model speech FP
# ──────────────────────────────────────────────────────────────────────


def phase_a(device) -> Tuple[bool, Dict]:
    print("=" * 58)
    print("Claim 1: base model speech FPR ~72%")
    print("=" * 58)

    model = load_model(MODEL_SAVE_PATH, device)
    ds = load_test_dataset('test')
    speech_idx = safe_indices_by_prefix(ds, 'as_speech_')
    if not speech_idx:
        print("  ERROR: no as_speech_*.pt found in prepared_data/test/0_safe/")
        return False, {}

    probs, _, _ = run_inference(model, ds, device, indices=speech_idx)
    speech_fpr = float((probs >= 0.5).mean())

    ok = BASE_SPEECH_FPR_LOW <= speech_fpr <= BASE_SPEECH_FPR_HIGH
    tag = "PASS" if ok else "FAIL"
    print(f"  speech samples:         {len(speech_idx)}")
    print(f"  speech FPR @ thr=0.50:  {speech_fpr:.3f}   "
          f"[{tag}  {BASE_SPEECH_FPR_LOW:.2f} ≤ fpr ≤ {BASE_SPEECH_FPR_HIGH:.2f}]")
    return ok, {'speech_fpr_base': speech_fpr,
                'n_speech': len(speech_idx)}


# ──────────────────────────────────────────────────────────────────────
# Deterministic 80/20 split of .wav basenames
# ──────────────────────────────────────────────────────────────────────


def sha256_bucket_80_20(filename: str) -> str:
    """Return 'fit' (80 %) or 'held_out' (20 %) deterministically.

    Uses the same sha256-of-basename approach as safecommute.utils.sha256_split
    (70/15/15) but rebalances to 80/20 for small per-site sets where 15 %
    held-out is too few samples. Filename-based so all chunks from one
    source land in the same bucket — identical guarantee to sha256_split.
    """
    h = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
    return 'held_out' if h >= 80 else 'fit'


def split_ambient_dir(ambient_dir: str) -> Tuple[List[str], List[str]]:
    """Split ambient-dir wavs 80/20 by basename hash. Returns (fit, held_out)
    lists of basenames (not full paths)."""
    if not os.path.isdir(ambient_dir):
        raise FileNotFoundError(f"ambient dir not found: {ambient_dir}")
    wavs = sorted(f for f in os.listdir(ambient_dir) if f.endswith('.wav'))
    fit, held = [], []
    for w in wavs:
        (held if sha256_bucket_80_20(w) == 'held_out' else fit).append(w)
    return fit, held


# ──────────────────────────────────────────────────────────────────────
# Phase B — ensure fine-tuned checkpoint
# ──────────────────────────────────────────────────────────────────────


def _stage_fit_subset(ambient_dir: str,
                      fit_basenames: List[str]) -> str:
    """Create a temp dir containing symlinks to the 80% fit wavs.

    finetune.py's --ambient-dir must be a directory; rather than copy
    58 wavs we symlink. Deterministic path so re-runs find the same
    staged dir.
    """
    stage = os.path.join(REPO_ROOT, '.scratch', 'ambient_fit')
    if os.path.exists(stage):
        import shutil
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    for name in fit_basenames:
        src = os.path.abspath(os.path.join(ambient_dir, name))
        dst = os.path.join(stage, name)
        try:
            os.symlink(src, dst)
        except OSError:
            # Windows without dev mode — fallback to copy.
            import shutil
            shutil.copy2(src, dst)
    return stage


def phase_b_ensure_model(environment: str,
                         ambient_dir: str,
                         held_out_override: Optional[str],
                         skip_finetune: bool) -> Tuple[bool, Dict]:
    print()
    print("=" * 58)
    print(f"Phase B: ensure fine-tuned '{environment}' checkpoint")
    print("=" * 58)

    model_path = f"models/{environment}_model.pth"
    thresh_path = f"models/{environment}_thresholds.json"
    have_model = os.path.exists(model_path)
    have_thresh = os.path.exists(thresh_path)
    if have_model and have_thresh:
        print(f"  Found {model_path}")
        print(f"  Found {thresh_path}")
        print("  Skipping fine-tune (checkpoints already present).")
        return True, {'model_path': model_path, 'thresh_path': thresh_path,
                      'fit_basenames': None, 'held_out_basenames': None,
                      'used_override': held_out_override is not None}

    if skip_finetune:
        print(f"  ERROR: --skip-finetune set but checkpoints missing:")
        print(f"    {model_path}  exists={have_model}")
        print(f"    {thresh_path}  exists={have_thresh}")
        return False, {}

    if not os.path.isdir(ambient_dir):
        print(f"  ERROR: ambient audio dir not found: {ambient_dir}")
        return False, {}

    # Decide the fine-tune set. If a separate held-out dir is provided,
    # fine-tune on the full ambient-dir. Otherwise split it 80/20.
    if held_out_override:
        if not os.path.isdir(held_out_override):
            print(f"  ERROR: held-out dir not found: {held_out_override}")
            return False, {}
        wavs = sorted(f for f in os.listdir(ambient_dir) if f.endswith('.wav'))
        fit_basenames = wavs
        held_basenames = sorted(
            f for f in os.listdir(held_out_override) if f.endswith('.wav'))
        stage = ambient_dir
        print(f"  Fine-tune set:   {len(fit_basenames)} wavs  "
              f"(all of {ambient_dir})")
        print(f"  Held-out set:    {len(held_basenames)} wavs  "
              f"(from {held_out_override})")
    else:
        fit_basenames, held_basenames = split_ambient_dir(ambient_dir)
        stage = _stage_fit_subset(ambient_dir, fit_basenames)
        print(f"  80/20 sha256 split of {ambient_dir}:")
        print(f"    fit:      {len(fit_basenames)} wavs → {stage}")
        print(f"    held-out: {len(held_basenames)} wavs (in {ambient_dir})")

    print(f"  Running finetune.py (this takes minutes)...")
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    cmd = [
        sys.executable, 'safecommute/pipeline/finetune.py',
        '--environment', environment,
        '--ambient-dir', stage,
        '--freeze-cnn',
    ]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: finetune.py failed with exit code {e.returncode}")
        return False, {}

    ok = os.path.exists(model_path) and os.path.exists(thresh_path)
    return ok, {
        'model_path': model_path,
        'thresh_path': thresh_path,
        'fit_basenames': fit_basenames,
        'held_out_basenames': held_basenames,
        'used_override': held_out_override is not None,
        'ambient_dir': ambient_dir,
        'held_out_dir': held_out_override or ambient_dir,
    }


# ──────────────────────────────────────────────────────────────────────
# Phase C — deployment measurement using test_deployment.py helpers
# ──────────────────────────────────────────────────────────────────────


def phase_c_measure(environment: str,
                    ambient_dir: str,
                    held_out_override: Optional[str],
                    threat_dir: str,
                    held_out_basenames: Optional[List[str]],
                    device) -> Tuple[bool, Dict]:
    print()
    print("=" * 58)
    print("Claim 2: fine-tuned overall FP ≤ 5% on HELD-OUT SITE AMBIENT")
    print("=" * 58)

    # Import the acceptance-test helpers directly — single source of truth
    # for the sliding-window + energy-gate + thresholding logic.
    from safecommute.pipeline.test_deployment import (
        load_model_and_stats, test_false_positive, test_threat_detection,
        test_consistency, test_silence,
    )

    model_path = f"models/{environment}_model.pth"
    thresh_path = f"models/{environment}_thresholds.json"
    if not (os.path.exists(model_path) and os.path.exists(thresh_path)):
        print(f"  ERROR: missing {model_path} or {thresh_path}")
        return False, {}

    with open(thresh_path) as f:
        thresholds = json.load(f)
    low_fpr = float(thresholds['low_fpr'])

    model, mean, std = load_model_and_stats(model_path)

    # Stage the held-out ambient so test_deployment only sees those wavs.
    if held_out_override and os.path.isdir(held_out_override):
        held_out_dir = held_out_override
        n_held = len([f for f in os.listdir(held_out_dir) if f.endswith('.wav')])
        print(f"  held-out ambient: {held_out_dir} ({n_held} wavs)")
    else:
        # Stage the 20% split from ambient-dir into a dedicated dir.
        if not held_out_basenames:
            # Recompute split if Phase B skipped.
            _, held_out_basenames = split_ambient_dir(ambient_dir)
        held_out_dir = os.path.join(REPO_ROOT, '.scratch', 'ambient_held_out')
        if os.path.exists(held_out_dir):
            import shutil
            shutil.rmtree(held_out_dir)
        os.makedirs(held_out_dir, exist_ok=True)
        for name in held_out_basenames:
            src = os.path.abspath(os.path.join(ambient_dir, name))
            dst = os.path.join(held_out_dir, name)
            try:
                os.symlink(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        n_held = len(held_out_basenames)
        print(f"  held-out ambient (staged 20 % split): {held_out_dir} "
              f"({n_held} wavs)")

    print(f"  low_fpr threshold:      {low_fpr:.3f}")

    # ── FP on held-out site ambient ─────────────────────────────────────
    fp_passed, fp_detail = test_false_positive(
        model, mean, std, held_out_dir, low_fpr, verbose=False)
    fp_rate = _parse_rate(fp_detail)
    print(f"  FP on held-out ambient:   {fp_rate:.3f}   "
          f"[{'PASS' if fp_rate is not None and fp_rate <= POST_FT_FP_MAX else 'FAIL'}"
          f"  ≤ {POST_FT_FP_MAX}]   ({fp_detail})")

    # ── Threat recall ───────────────────────────────────────────────────
    tr_passed, tr_detail = test_threat_detection(
        model, mean, std, threat_dir, low_fpr, verbose=False)
    tr_rate = _parse_rate(tr_detail)
    print(f"  Threat recall on {threat_dir}: {tr_rate:.3f}   "
          f"[{'PASS' if tr_rate is not None and tr_rate >= POST_FT_THREAT_RECALL_MIN else 'FAIL'}"
          f"  ≥ {POST_FT_THREAT_RECALL_MIN}]   ({tr_detail})")

    # ── Speech-FP post-fine-tune (universal speech subset as proxy) ────
    ds = load_test_dataset('test')
    speech_idx = safe_indices_by_prefix(ds, 'as_speech_')
    model_dev = load_model(model_path, device)
    sp_probs, _, _ = run_inference(model_dev, ds, device, indices=speech_idx)
    speech_fp_post = float((sp_probs >= low_fpr).mean())
    print(f"  Speech FP post-FT:        {speech_fp_post:.3f}   "
          f"[{'PASS' if speech_fp_post <= POST_FT_SPEECH_FP_MAX else 'WARN'}"
          f"  ≤ {POST_FT_SPEECH_FP_MAX}]  "
          f"(universal as_speech subset, n={len(speech_idx)})")

    # ── Determinism + silence gating ────────────────────────────────────
    det_passed, det_detail = test_consistency(model)
    sil_passed, sil_detail = test_silence(model, mean, std)
    print(f"  Determinism:              "
          f"{'PASS' if det_passed else 'FAIL'}  ({det_detail})")
    print(f"  Silence gating:           "
          f"{'PASS' if sil_passed else 'FAIL'}  ({sil_detail})")

    all_ok = (bool(fp_passed) and bool(tr_passed)
              and bool(det_passed) and bool(sil_passed)
              and speech_fp_post <= POST_FT_SPEECH_FP_MAX)

    blob = {
        'site': environment,
        'model_path': model_path,
        'thresholds_file': thresh_path,
        'low_fpr_threshold': low_fpr,
        'ambient_fit_dir': ambient_dir,
        'held_out_dir': held_out_dir,
        'held_out_wavs': n_held,
        'threat_dir': threat_dir,
        'fp_rate': fp_rate,
        'threat_recall': tr_rate,
        'speech_fp_post': speech_fp_post,
        'determinism_ok': bool(det_passed),
        'silence_gate_ok': bool(sil_passed),
        'fp_passed': bool(fp_passed),
        'threat_passed': bool(tr_passed),
        'fp_detail': fp_detail,
        'threat_detail': tr_detail,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f'phase_b_{environment}.json')
    with open(out_path, 'w') as f:
        json.dump(blob, f, indent=2)
    print(f"\n  Phase B blob → {out_path}")
    return all_ok, blob


def _parse_rate(detail: str) -> Optional[float]:
    """Pull the first "NN.N%" number out of a test_deployment detail string."""
    import re
    m = re.search(r'(\d+\.\d+)%', detail)
    return float(m.group(1)) / 100.0 if m else None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['a', 'b', 'c', 'all'], default='all',
                   help="'a' = base-only; 'c' = deployment only; 'all' = both")
    p.add_argument('--skip-finetune', action='store_true')
    p.add_argument('--environment', default='metro',
                   help="Site name — produces models/<env>_model.pth")
    p.add_argument('--ambient-dir', default='raw_data/youtube_metro',
                   help='Ambient audio used for fine-tuning')
    p.add_argument('--held-out-ambient-dir', default=None,
                   help='Held-out ambient for Phase C measurement. '
                        'If not set, Phase C uses a sha256 20 % split of '
                        '--ambient-dir.')
    p.add_argument('--threat-dir', default='raw_data/youtube_screams',
                   help='Real threat .wav files for recall measurement.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"SafeCommute AI — FP-rate claim validation  (device={device})\n")

    a_ok, a_info = phase_a(device)
    if args.phase == 'a':
        print()
        print("Result:", "CLAIM 1 VALIDATED" if a_ok else "CLAIM 1 FAILED")
        sys.exit(0 if a_ok else 1)

    b_ok, b_info = phase_b_ensure_model(
        args.environment, args.ambient_dir,
        args.held_out_ambient_dir, args.skip_finetune)
    if not b_ok:
        print("\nResult: CANNOT VALIDATE CLAIM 2 (fine-tuned model unavailable)")
        sys.exit(1)
    if args.phase == 'b':
        print("\nResult: CHECKPOINT AVAILABLE")
        sys.exit(0)

    c_ok, c_blob = phase_c_measure(
        args.environment,
        args.ambient_dir,
        args.held_out_ambient_dir,
        args.threat_dir,
        b_info.get('held_out_basenames'),
        device)

    print()
    print("=" * 58)
    any_fail = (not a_ok) or (not c_ok)
    if not any_fail:
        print("Result: ALL CLAIMS VALIDATED")
        sys.exit(0)
    print("Result:", "CLAIM 1 FAILED" if not a_ok else "CLAIM 1 ok;",
          "CLAIM 2 FAILED" if not c_ok else "CLAIM 2 ok")
    sys.exit(1)


if __name__ == '__main__':
    main()
