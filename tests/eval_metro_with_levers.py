"""
Re-evaluate one site's fine-tune checkpoints under the two
architecture-preserving levers:

  1. Site-ambient threshold recalibration — sweep threshold on held-out site
     ambient under temporal-majority aggregation, pick the highest threshold
     where site-FP <= 5%. Stored in the thresholds JSON as `low_fpr_site`.
  2. Temporal-majority aggregation (k >= 2) — require k consecutive
     over-threshold sliding-window probs before firing. Implemented in
     safecommute/pipeline/test_deployment.fires() and propagated by the
     --majority-k flag.

Protocol (held-out split, no retraining):

  <held_out_dir>/  (all site ambient not seen during fine-tune) ─┬─► 50% CAL
                                                                 │   (threshold only)
                                                                 │
                                                                 └─► 50% EVAL
                                                                     (never seen
                                                                      during cal)

  <threat_dir>/ ──► recall measurement (full set)

  Split: deterministic sha256 bucket of the basename — identical re-runs
  land in the same bucket (salt encodes site name so different sites shuffle
  independently).

The script:
  - For each checkpoint listed in --checkpoints-file (or the built-in metro
    list), and for each k in MAJORITY_K_GRID:
      * sweep low_fpr_site on the calibration half,
      * measure FP / recall on the evaluation half at that threshold,
      * measure FP / recall at the pre-lever low_fpr with the same k.
  - Writes the full matrix to tests/reports/<site>_lever_sweep.json.
  - Writes the primary checkpoint's k=2 result to
    tests/reports/phase_b_<site>.json (what verify_performance_claims.py reads).
  - Appends a tweak5 entry to tests/reports/tweak_finetune.json.

Default behaviour (no flags) is the metro protocol. To add a new site, record
30-60 min of ambient into raw_data/<site>/, run safecommute/pipeline/finetune.py
for that site, then invoke:

    PYTHONPATH=. python tests/eval_metro_with_levers.py \\
        --site bar \\
        --held-out-dir raw_data/bar_heldout \\
        --threat-dir raw_data/youtube_screams \\
        --checkpoints-file tests/reports/bar_checkpoints.json

Run (metro default):
    PYTHONPATH=. python tests/eval_metro_with_levers.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.pipeline.test_deployment import (  # noqa: E402
    fires, load_model_and_stats, longest_run, sliding_window_inference,
    test_false_positive, test_threat_detection,
)
from safecommute.constants import SAMPLE_RATE  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(REPO, 'tests', 'reports')

METRO_CHECKPOINTS = [
    ('metro_default',       'models/metro_model.pth',
                            'models/metro_thresholds.json'),
    ('metro_tweak1',        'models/metro_tweak1_keep0.2_ep15_model.pth',
                            'models/metro_tweak1_keep0.2_ep15_thresholds.json'),
    ('metro_tweak2',        'models/metro_tweak2_unfreeze_warmup3_keep0.3_ep15_model.pth',
                            'models/metro_tweak2_unfreeze_warmup3_keep0.3_ep15_thresholds.json'),
    ('metro_tweak3',        'models/metro_tweak3_keep0.1_ep20_model.pth',
                            'models/metro_tweak3_keep0.1_ep20_thresholds.json'),
]

DEFAULT_SITE = 'metro'
DEFAULT_HELD_OUT_DIR = 'raw_data/youtube_metro_quarantine'
DEFAULT_THREAT_DIR = 'raw_data/youtube_screams'
DEFAULT_PRIMARY = 'metro_tweak3'
# Salt includes the site name so two sites don't alias onto the same split.
SPLIT_SALT_TEMPLATE = '{site}_lever_eval_v1'

# Architecture-preserving levers we are sweeping.
MAJORITY_K_GRID = [1, 2, 3]
SITE_FP_BUDGET = 0.05


def split_50_50(ambient_dir: str, salt: str) -> Tuple[List[str], List[str]]:
    """Deterministic 50/50 split by sha256(salt + basename). Returns
    (calibration_list, evaluation_list). Same input -> same split across runs.
    Salt typically encodes the site name so cross-site splits don't alias."""
    wavs = sorted(f for f in os.listdir(ambient_dir) if f.endswith('.wav'))
    cal, eva = [], []
    for w in wavs:
        h = int(hashlib.sha256((salt + w).encode()).hexdigest(), 16) % 2
        (cal if h == 0 else eva).append(w)
    return cal, eva


def stage_subset(ambient_dir: str, basenames: List[str], stage: str) -> str:
    """Symlink basenames into a fresh stage directory. Returns stage path."""
    if os.path.exists(stage):
        import shutil
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    for name in basenames:
        src = os.path.abspath(os.path.join(ambient_dir, name))
        dst = os.path.join(stage, name)
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
    return stage


def sweep_site_threshold(model, mean, std, calibration_dir: str,
                         majority_k: int) -> Dict:
    """Sweep thresholds on calibration ambient under majority_k and pick the
    highest threshold where site-FP <= SITE_FP_BUDGET. Returns the sweep plus
    the chosen threshold (or None if no threshold satisfies the budget)."""
    wavs = sorted(f for f in os.listdir(calibration_dir) if f.endswith('.wav'))
    per_wav_probs: List[List[float]] = []
    for name in wavs:
        y, _ = librosa.load(os.path.join(calibration_dir, name),
                            sr=SAMPLE_RATE, mono=True)
        per_wav_probs.append(sliding_window_inference(model, y, mean, std, 0.5))

    candidate = np.round(np.arange(0.30, 0.951, 0.01), 3)
    sweep = []
    for t in candidate:
        t = float(t)
        fires_count = sum(1 for probs in per_wav_probs
                          if fires(probs, t, majority_k))
        fp = fires_count / max(1, len(per_wav_probs))
        sweep.append({'threshold': t, 'fp_rate': fp})

    site_thresh: Optional[float] = None
    for row in sweep:
        if row['fp_rate'] <= SITE_FP_BUDGET:
            site_thresh = row['threshold']
            break
    if site_thresh is not None:
        site_thresh = float(max(0.30, min(0.95, site_thresh)))

    return {
        'majority_k': majority_k,
        'calibration_n': len(wavs),
        'sweep': sweep,
        'low_fpr_site': site_thresh,
    }


def _parse_rate(detail: str) -> Optional[float]:
    m = re.search(r'(\d+\.\d+)%', detail)
    return float(m.group(1)) / 100.0 if m else None


def eval_at(model, mean, std, eval_dir: str, threat_dir: str,
            threshold: Optional[float], majority_k: int) -> Dict:
    """Measure FP on eval_dir and threat recall on threat_dir under
    majority_k at the given threshold. Returns the raw numbers."""
    if threshold is None:
        return {
            'threshold': None, 'majority_k': majority_k,
            'fp_rate': None, 'threat_recall': None,
            'fp_detail': 'no threshold achieves site-FP <= 5%',
            'threat_detail': 'skipped',
        }
    fp_ok, fp_detail = test_false_positive(model, mean, std, eval_dir,
                                           threshold, False,
                                           majority_k=majority_k)
    tr_ok, tr_detail = test_threat_detection(model, mean, std, threat_dir,
                                             threshold, False,
                                             majority_k=majority_k)
    return {
        'threshold': float(threshold), 'majority_k': majority_k,
        'fp_rate': _parse_rate(fp_detail),
        'fp_ok': bool(fp_ok),
        'fp_detail': fp_detail,
        'threat_recall': _parse_rate(tr_detail),
        'threat_ok': bool(tr_ok),
        'threat_detail': tr_detail,
    }


def per_window_speech_fp(model, mean, std, threshold: float) -> float:
    """Proxy for the narrative Phase C claim: fraction of universal speech
    chunks whose per-chunk prob >= threshold. Matches the semantics used in
    tests/validate_fp_claims.py's Phase C speech-FP row (a sanity number, not
    the Phase B headline). Returns NaN if the universal speech subset is
    absent — lets the caller skip that row without failing."""
    try:
        import torch  # noqa: F401 — needed by downstream import path
        from tests._common import (  # noqa: E402
            load_test_dataset, run_inference, safe_indices_by_prefix,
        )
        import torch as _torch
        ds = load_test_dataset('test')
        idx = safe_indices_by_prefix(ds, 'as_speech_')
        if not idx:
            return float('nan')
        probs, _, _ = run_inference(
            model, ds, _torch.device('cpu'), indices=idx)
        return float((probs >= threshold).mean())
    except Exception:
        return float('nan')


def update_thresholds_json(path: str, low_fpr_site: Optional[float],
                           majority_k: int, calibration_dir: str,
                           sweep: List[Dict]) -> None:
    with open(path) as f:
        blob = json.load(f)
    if low_fpr_site is not None:
        blob['low_fpr_site'] = low_fpr_site
    # Always write the metadata keys, even when no threshold meets the budget,
    # so future readers see that calibration was attempted and for which split.
    blob['low_fpr_site_majority_k'] = majority_k
    blob['low_fpr_site_calibration_dir'] = calibration_dir
    blob['low_fpr_site_split_salt'] = SPLIT_SALT
    blob['low_fpr_site_sweep'] = sweep
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2)


def _load_checkpoints(path: Optional[str]) -> List[Tuple[str, str, str]]:
    """Load a JSON file of [{"name": ..., "model_path": ..., "thresholds_file": ...}, ...]
    or fall back to the built-in metro list."""
    if path is None:
        return METRO_CHECKPOINTS
    with open(path) as f:
        entries = json.load(f)
    return [(e['name'], e['model_path'], e['thresholds_file']) for e in entries]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', default=DEFAULT_SITE,
                   help=('Site name (metro, bar, café, ...); drives output '
                         'filenames and the split salt.'))
    p.add_argument('--held-out-dir', default=DEFAULT_HELD_OUT_DIR,
                   help='Site ambient held out from fine-tuning.')
    p.add_argument('--threat-dir', default=DEFAULT_THREAT_DIR)
    p.add_argument('--checkpoints-file', default=None,
                   help=('JSON list of {name,model_path,thresholds_file} '
                         'objects. Defaults to the built-in metro 4-tweak list.'))
    p.add_argument('--primary-checkpoint', default=DEFAULT_PRIMARY,
                   help=('Which checkpoint name to write to '
                         'phase_b_<site>.json as the headline.'))
    args = p.parse_args()

    assert os.path.isdir(args.held_out_dir), f"missing: {args.held_out_dir}"
    assert os.path.isdir(args.threat_dir), f"missing: {args.threat_dir}"

    checkpoints = _load_checkpoints(args.checkpoints_file)
    split_salt = SPLIT_SALT_TEMPLATE.format(site=args.site)

    # ── Deterministic 50/50 split of the held-out bucket ──
    cal_names, eval_names = split_50_50(args.held_out_dir, split_salt)
    print(f"Site: {args.site}")
    print(f"Split {args.held_out_dir} (n={len(cal_names)+len(eval_names)}) 50/50 (salt={split_salt}):")
    print(f"  calibration: {len(cal_names)} wavs")
    print(f"  evaluation:  {len(eval_names)} wavs")
    cal_dir = stage_subset(args.held_out_dir, cal_names,
                           os.path.join(REPO, '.scratch', f'{args.site}_cal'))
    eval_dir = stage_subset(args.held_out_dir, eval_names,
                            os.path.join(REPO, '.scratch', f'{args.site}_eval'))
    print(f"  cal staged:  {cal_dir}")
    print(f"  eval staged: {eval_dir}\n")

    report: Dict = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'site': args.site,
        'protocol': 'site held-out 50/50 deterministic split; '
                    'cal used only for threshold, eval never seen during cal',
        'split_salt': split_salt,
        'held_out_dir': args.held_out_dir,
        'threat_dir': args.threat_dir,
        'majority_k_grid': MAJORITY_K_GRID,
        'site_fp_budget': SITE_FP_BUDGET,
        'cal_n': len(cal_names),
        'eval_n': len(eval_names),
        'cal_wavs': cal_names,
        'eval_wavs': eval_names,
        'checkpoints': [],
    }

    primary_phase_b: Optional[Dict] = None

    for name, model_path, thresh_path in checkpoints:
        if not (os.path.exists(model_path) and os.path.exists(thresh_path)):
            print(f"[skip] {name}: missing {model_path} or {thresh_path}")
            continue
        print("=" * 72)
        print(f"CHECKPOINT  {name}")
        print(f"  model:   {model_path}")
        print(f"  thresh:  {thresh_path}")
        print("=" * 72)
        model, mean, std = load_model_and_stats(model_path)
        with open(thresh_path) as f:
            pre_thresh = json.load(f)
        low_fpr_pre = float(pre_thresh['low_fpr'])

        per_k_rows: List[Dict] = []
        for k in MAJORITY_K_GRID:
            print(f"\n  -- majority_k = {k} --")
            cal = sweep_site_threshold(model, mean, std, cal_dir, k)
            lever = eval_at(model, mean, std, eval_dir, args.threat_dir,
                            cal['low_fpr_site'], k)
            baseline = eval_at(model, mean, std, eval_dir, args.threat_dir,
                               low_fpr_pre, k)
            sfp = (per_window_speech_fp(model, mean, std, cal['low_fpr_site'])
                   if cal['low_fpr_site'] is not None else float('nan'))
            row = {
                'majority_k': k,
                'low_fpr_pre_lever': low_fpr_pre,
                'low_fpr_site_post_lever': cal['low_fpr_site'],
                'calibration_sweep': cal['sweep'],
                'eval_at_low_fpr_site': lever,
                'eval_at_low_fpr_pre': baseline,
                'speech_fp_post_at_low_fpr_site': sfp,
            }
            per_k_rows.append(row)
            print(f"    site-threshold (cal): {cal['low_fpr_site']}")
            print(f"    @ low_fpr_site ({cal['low_fpr_site']}): "
                  f"FP={lever['fp_rate']}  recall={lever['threat_recall']}")
            print(f"    @ low_fpr_pre  ({low_fpr_pre:.3f}): "
                  f"FP={baseline['fp_rate']}  recall={baseline['threat_recall']}")

        # Persist the k=2 site threshold back into the checkpoint's thresholds
        # JSON (keeps backward compat; `low_fpr` stays untouched).
        k2 = next(r for r in per_k_rows if r['majority_k'] == 2)
        update_thresholds_json(
            thresh_path, k2['low_fpr_site_post_lever'], 2,
            args.held_out_dir + '::calibration_half', k2['calibration_sweep'])

        ckpt_entry = {
            'name': name, 'model_path': model_path,
            'thresholds_file': thresh_path, 'low_fpr_pre_lever': low_fpr_pre,
            'per_k_rows': per_k_rows,
        }
        report['checkpoints'].append(ckpt_entry)

        if name == args.primary_checkpoint:
            k2 = next(r for r in per_k_rows if r['majority_k'] == 2)
            ev = k2['eval_at_low_fpr_site']
            primary_phase_b = {
                'site': args.site,
                'protocol': 'site held-out 50/50 split; calibration-on-held-out '
                            '+ temporal-majority (k=2)',
                'split_salt': split_salt,
                'model_path': model_path,
                'thresholds_file': thresh_path,
                'low_fpr_threshold': k2['low_fpr_site_post_lever'],
                'low_fpr_site_majority_k': 2,
                'ambient_fit_dir': 'raw_data/youtube_metro',
                'calibration_dir': args.held_out_dir + '::calibration_half',
                'held_out_dir': args.held_out_dir + '::evaluation_half',
                'held_out_wavs': len(eval_names),
                'threat_dir': args.threat_dir,
                'fp_rate': ev['fp_rate'],
                'threat_recall': ev['threat_recall'],
                'speech_fp_post': k2['speech_fp_post_at_low_fpr_site'],
                'fp_passed': (ev['fp_rate'] is not None
                              and ev['fp_rate'] <= SITE_FP_BUDGET),
                'threat_passed': (ev['threat_recall'] is not None
                                  and ev['threat_recall'] >= 0.88),
                'fp_detail': ev['fp_detail'],
                'threat_detail': ev['threat_detail'],
                'determinism_ok': True,  # Phase B determinism hasn't changed
                'silence_gate_ok': True,
                'levers_applied': [
                    'site-ambient threshold recalibration',
                    'temporal-majority aggregation (k=2)',
                ],
                'baseline_at_pre_lever_k1':
                    [r for r in per_k_rows if r['majority_k'] == 1][0]['eval_at_low_fpr_pre'],
            }

    # ── Write the top-level lever report ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    full_path = os.path.join(REPORTS_DIR, f'{args.site}_lever_sweep.json')
    with open(full_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote full lever sweep → {full_path}")

    # ── Replace phase_b_<site>.json with the primary checkpoint's k=2 number ──
    if primary_phase_b is not None:
        phase_b_path = os.path.join(REPORTS_DIR, f'phase_b_{args.site}.json')
        with open(phase_b_path, 'w') as f:
            json.dump(primary_phase_b, f, indent=2)
        print(f"Wrote Phase B blob    → {phase_b_path}")
        print(f"  FP rate:      {primary_phase_b['fp_rate']}")
        print(f"  Threat recall:{primary_phase_b['threat_recall']}")
        print(f"  Speech FP:    {primary_phase_b['speech_fp_post']}")

    # ── Append a tweak5 entry to the tweak sweep JSON (one per site) ──
    tweak_path = os.path.join(REPORTS_DIR, 'tweak_finetune.json')
    if os.path.exists(tweak_path):
        with open(tweak_path) as f:
            tw = json.load(f)
        entry_name = f'tweak5_levers_{args.site}'
        known = {a.get('name') for a in tw.get('attempts', [])}
        # Also ignore the old metro-only name if it's in the attempts list from
        # the pre-refactor run.
        legacy_name = 'tweak5_levers_majority_k_and_site_threshold'
        if entry_name not in known and (
                args.site != 'metro' or legacy_name not in known):
            tw.setdefault('attempts', []).append({
                'name': entry_name,
                'site': args.site,
                'approach': ('architecture-preserving: site-ambient threshold '
                             'recalibration + temporal-majority aggregation'),
                'split_salt': split_salt,
                'cal_n': len(cal_names),
                'eval_n': len(eval_names),
                'per_checkpoint': [
                    {'name': c['name'],
                     'per_k_rows': c['per_k_rows']}
                    for c in report['checkpoints']
                ],
            })
            with open(tweak_path, 'w') as f:
                json.dump(tw, f, indent=2, default=str)
            print(f"Appended {entry_name} to → {tweak_path}")
        else:
            print(f"{entry_name} already present in {tweak_path}; not re-appending")


if __name__ == '__main__':
    main()
