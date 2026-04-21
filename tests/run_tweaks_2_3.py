"""
Run the remaining tweaks (1-measurement, 2, 3) incrementally.

Unlike tests/tweak_finetune.py this writes to tests/reports/tweak_finetune.json
*after each attempt*, so an interrupted run still leaves partial results.
`python -u` recommended.

Run:
    PYTHONPATH=. python -u tests/run_tweaks_2_3.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.tweak_finetune import _measure_on_heldout, _recalibrate_threshold_on_heldout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(REPO, 'tests', 'reports', 'tweak_finetune.json')
AMBIENT = 'raw_data/youtube_metro'
HELD_OUT = 'raw_data/youtube_metro_quarantine'
THREATS = 'raw_data/youtube_screams'


def load_sweep() -> Dict:
    with open(OUT) as f:
        return json.load(f)


def save_sweep(j: Dict) -> None:
    with open(OUT, 'w') as f:
        json.dump(j, f, indent=2)
    print(f"  [wrote {OUT}]", flush=True)


def measure_existing(env_name: str, label: str, ratio: float, epochs: int,
                     warmup_epochs: int, freeze_cnn: bool) -> Dict:
    """Measure an already-saved checkpoint on held-out."""
    t0 = time.time()
    print(f"\n== MEASURE {label} ==", flush=True)
    meas = _measure_on_heldout(env_name, HELD_OUT, THREATS)
    print(f"  default: FP {meas['fp_rate']*100:.1f}%  "
          f"recall {meas['threat_recall']*100:.1f}%  "
          f"thr {meas['threshold_used']:.3f}", flush=True)
    recal = _recalibrate_threshold_on_heldout(env_name, HELD_OUT, THREATS)
    best = recal['best_fp_le_5pct']
    if best:
        print(f"  recal: thr={best['threshold']:.3f}  "
              f"FP={best['fp_rate']*100:.1f}%  "
              f"recall={best['threat_recall']*100:.1f}%", flush=True)
    else:
        print("  recal: no threshold achieves FP ≤ 5%", flush=True)
    dt = time.time() - t0
    print(f"  [elapsed {dt:.1f}s]", flush=True)
    return {
        'name': label,
        'keep_safe_ratio': ratio,
        'epochs': epochs,
        'warmup_epochs': warmup_epochs,
        'freeze_cnn': freeze_cnn,
        'measured_default': meas,
        'threshold_recal': recal,
    }


def run_tweak(env_name: str, label: str, ratio: float, epochs: int,
              warmup_epochs: int, freeze_cnn: bool) -> Optional[Dict]:
    """Train then measure."""
    model_path = f"models/{env_name}_model.pth"
    if os.path.exists(model_path):
        print(f"  [{label}] checkpoint exists, skipping retrain", flush=True)
    else:
        print(f"\n== TRAIN {label} ==", flush=True)
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        env['PYTHONUNBUFFERED'] = '1'
        cmd = [
            sys.executable, '-u', 'safecommute/pipeline/finetune.py',
            '--environment', env_name,
            '--ambient-dir', AMBIENT,
            '--epochs', str(epochs),
            '--lr', '1e-4',
            '--keep-safe-ratio', str(ratio),
            '--warmup-epochs', str(warmup_epochs),
        ]
        if freeze_cnn:
            cmd.append('--freeze-cnn')
        print(f"  $ {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, cwd=REPO, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FINETUNE FAILED exit={e.returncode}", flush=True)
            return None
    return measure_existing(env_name, label, ratio, epochs,
                            warmup_epochs, freeze_cnn)


def main():
    j = load_sweep()
    known_names = {a.get('name') for a in j.get('attempts', [])
                   if a.get('measured_default') is not None}

    # ── Tweak 1: measure only (checkpoint already saved) ───────────────
    name1 = 'tweak1_keep0.2_ep15'
    if name1 in known_names:
        print(f"SKIP {name1} (already have measurement)", flush=True)
    else:
        r = measure_existing(f'metro_{name1}', name1, 0.2, 15, 0, True)
        j['attempts'].append(r)
        save_sweep(j)

    # ── Tweak 2: unfreeze CNN with warmup ──────────────────────────────
    name2 = 'tweak2_unfreeze_warmup3_keep0.3_ep15'
    if name2 in known_names:
        print(f"SKIP {name2}", flush=True)
    else:
        r = run_tweak(f'metro_{name2}', name2,
                      ratio=0.3, epochs=15, warmup_epochs=3, freeze_cnn=False)
        if r is not None:
            j['attempts'].append(r)
            save_sweep(j)

    # ── Tweak 3: dominant-metro, keep 0.1, 20 epochs ───────────────────
    name3 = 'tweak3_keep0.1_ep20'
    if name3 in known_names:
        print(f"SKIP {name3}", flush=True)
    else:
        r = run_tweak(f'metro_{name3}', name3,
                      ratio=0.1, epochs=20, warmup_epochs=0, freeze_cnn=True)
        if r is not None:
            j['attempts'].append(r)
            save_sweep(j)

    # ── Tweak 4: recalibrate low_fpr using the FIT metro ambient ───────
    # This doesn't retrain anything — uses the metro_model.pth but picks
    # a threshold that targets ≤5% FP on the FIT set (not the universal
    # prepared_data/test). This is the most principled threshold recipe.
    name4 = 'tweak4_site_threshold_recal'
    if name4 not in known_names:
        print(f"\n== TWEAK 4: site-ambient threshold recalibration ==",
              flush=True)
        t0 = time.time()
        recal = _recalibrate_threshold_on_heldout(
            'metro', HELD_OUT, THREATS, target_fp=0.05)
        best = recal['best_fp_le_5pct']
        if best:
            print(f"  best: thr={best['threshold']:.3f}  "
                  f"FP={best['fp_rate']*100:.1f}%  "
                  f"recall={best['threat_recall']*100:.1f}%", flush=True)
        print(f"  [elapsed {time.time()-t0:.1f}s]", flush=True)
        j['attempts'].append({
            'name': name4,
            'approach': 'recalibrate low_fpr on held-out (metro_model.pth)',
            'threshold_recal': recal,
        })
        save_sweep(j)

    print("\n== DONE ==", flush=True)
    print(f"attempts written: {len(j['attempts'])}", flush=True)
    for a in j['attempts']:
        md = a.get('measured_default')
        if md:
            print(f"  {a['name']}: FP {md['fp_rate']*100:.1f}%  "
                  f"recall {md['threat_recall']*100:.1f}%", flush=True)


if __name__ == '__main__':
    main()
