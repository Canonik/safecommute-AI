"""
Final release gate: compute artefact hashes, collate verifier output,
write tests/reports/SUMMARY.md and tests/reports/artefacts.sha256.

Run once after all other Steps are green:
    PYTHONPATH=. python tests/verify_performance_claims.py   # exit 0
    PYTHONPATH=. python tests/finalize_release.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS = os.path.join(REPO_ROOT, 'tests', 'reports')


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def main():
    os.makedirs(REPORTS, exist_ok=True)

    # ── Artefact hashes ─────────────────────────────────────────────────
    targets = sorted(set(
        glob.glob(os.path.join(REPO_ROOT, 'models', '*.pth')) +
        glob.glob(os.path.join(REPO_ROOT, 'models', '*.onnx')) +
        glob.glob(os.path.join(REPO_ROOT, 'models', '*.onnx.data'))  # flagged if leaked
    ))
    lines = []
    hashes = {}
    for t in targets:
        rel = os.path.relpath(t, REPO_ROOT)
        h = sha256_file(t)
        size_mb = os.path.getsize(t) / (1024 ** 2)
        lines.append(f"{h}  {rel}  ({size_mb:.2f} MB)")
        hashes[rel] = {'sha256': h, 'size_mb': size_mb}
    with open(os.path.join(REPORTS, 'artefacts.sha256'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # ── Verifier summary ────────────────────────────────────────────────
    vp = os.path.join(REPORTS, 'verify_performance_claims.json')
    verifier = None
    if os.path.exists(vp):
        with open(vp) as f:
            verifier = json.load(f)

    # ── Phase B summary ─────────────────────────────────────────────────
    phase_b = {}
    for f_ in glob.glob(os.path.join(REPORTS, 'phase_b_*.json')):
        with open(f_) as g:
            phase_b[os.path.basename(f_)] = json.load(g)

    # ── Tweak sweep summary ─────────────────────────────────────────────
    tweak = None
    tw = os.path.join(REPORTS, 'tweak_finetune.json')
    if os.path.exists(tw):
        with open(tw) as f:
            tweak = json.load(f)

    # ── End-to-end latency ──────────────────────────────────────────────
    e2e = None
    e2e_path = os.path.join(REPORTS, 'measure_e2e_latency.json')
    if os.path.exists(e2e_path):
        with open(e2e_path) as f:
            e2e = json.load(f)

    # ── SUMMARY.md ──────────────────────────────────────────────────────
    git_sha = _git_sha()
    now = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())

    md = [f"# SafeCommute AI — Release Summary ({now})", ""]
    md.append(f"**Git HEAD**: `{git_sha}`\n")

    # Verifier
    if verifier:
        s = verifier.get('summary', {})
        md.append("## Verifier")
        md.append(f"- Total rows: {s.get('total')}")
        md.append(f"- PASS: {s.get('passed')}")
        md.append(f"- FAIL (hard): {s.get('failed_hard')}")
        md.append(f"- WARN: {s.get('warned')}")
        hw = verifier.get('hw_disclosure', {})
        md.append(f"- Hardware: {hw.get('cpu_model', 'unknown')}, "
                  f"{hw.get('cores', '?')} cores, "
                  f"{hw.get('torch_threads', '?')} torch threads, "
                  f"governor={hw.get('governor', 'unknown')}")
        md.append(f"- torch {hw.get('torch_version')}, "
                  f"onnxruntime {hw.get('onnxruntime_version')}, "
                  f"BLAS={hw.get('blas')}")
        md.append("")

    # Phase A core numbers
    if verifier and 'phase_a_full_metrics' in verifier:
        m = verifier['phase_a_full_metrics']
        md.append("## Phase A (base model on prepared_data/test)")
        md.append(f"| Metric | Measured |")
        md.append(f"|---|---|")
        md.append(f"| AUC-ROC | {m['auc_roc']:.4f} |")
        md.append(f"| Accuracy @ 0.50 | {m['accuracy']:.4f} |")
        md.append(f"| F1 weighted @ 0.50 | {m['f1_weighted']:.4f} |")
        md.append(f"| Overall FPR @ 0.50 (leakage) | {m['overall_fpr']:.4f} |")
        md.append(f"| Micro threat recall | {m['overall_tpr']:.4f} |")
        md.append(f"| Speech FPR (as_speech) | {m['fpr_by_source'].get('as_speech', {}).get('rate', 'N/A'):.4f} |")
        md.append(f"| Laughter FPR (as_laughter) | {m['fpr_by_source'].get('as_laughter', {}).get('rate', 'N/A'):.4f} |")
        md.append("")

    # Phase B
    if phase_b:
        md.append("## Phase B (per-site fine-tune, held-out ambient)")
        md.append("| Site | FP | Recall | Speech-FP post | Determinism | Silence gate |")
        md.append("|---|---|---|---|---|---|")
        for name, b in phase_b.items():
            md.append(
                f"| {b.get('site', name)} | "
                f"{(b.get('fp_rate') or 0.0)*100:.1f}% (target ≤5%) | "
                f"{(b.get('threat_recall') or 0.0)*100:.1f}% (target ≥86%) | "
                f"{(b.get('speech_fp_post') or 0.0)*100:.1f}% (target ≤10%) | "
                f"{'PASS' if b.get('determinism_ok') else 'FAIL'} | "
                f"{'PASS' if b.get('silence_gate_ok') else 'FAIL'} |")
        md.append("")

    # Tweak sweep
    if tweak:
        md.append("## Tweak sweep (`tests/reports/tweak_finetune.json`)")
        md.append("| Config | FP default thr | Threat recall | Notes |")
        md.append("|---|---|---|---|")
        for a in tweak.get('attempts', []):
            mn = a.get('name', '?')
            md_row = a.get('measured_default') or {}
            fp = md_row.get('fp_rate')
            tr = md_row.get('threat_recall')
            thr = md_row.get('threshold_used')
            if fp is None:
                md.append(f"| {mn} | (no retrain measurement) | | see threshold-recal below |")
            else:
                md.append(
                    f"| {mn} | "
                    f"{fp*100:.1f}% @ thr={thr:.3f} | "
                    f"{tr*100:.1f}% | |")
        md.append("")

    # E2E latency
    if e2e:
        md.append("## End-to-end latency (preprocess + model)")
        md.append("| Run | median (ms) | p99 (ms) |")
        md.append("|---|---|---|")
        for name, r in (e2e.get('runs') or {}).items():
            if isinstance(r, dict) and 'median' in r:
                md.append(f"| {name} | {r['median']:.1f} | {r['p99']:.1f} |")
        md.append("")

    # Artefact hashes
    md.append("## Artefact hashes (`tests/reports/artefacts.sha256`)")
    md.append("| File | SHA256 | Size (MB) |")
    md.append("|---|---|---|")
    for rel, h in hashes.items():
        md.append(f"| `{rel}` | `{h['sha256'][:16]}…` | {h['size_mb']:.2f} |")
    md.append("")

    # Footer
    md.append("---")
    md.append(f"Generated by `tests/finalize_release.py` at {now}.")
    md.append("Re-run `python tests/verify_performance_claims.py` to refresh.")

    summary_path = os.path.join(REPORTS, 'SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(md) + '\n')

    print(f"SUMMARY → {summary_path}")
    print(f"artefacts.sha256 → {os.path.join(REPORTS, 'artefacts.sha256')}")
    print(f"{len(hashes)} artefacts hashed.")
    # If any .onnx.data sidecar exists, flag it loudly — Step 1 should have removed them.
    for rel in hashes:
        if rel.endswith('.onnx.data'):
            print(f"WARNING: {rel} is a sidecar — Step 1 (single-file ONNX fix) may have regressed.")


if __name__ == '__main__':
    main()
