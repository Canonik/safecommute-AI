"""
Generate synthetic silence/quiet samples for SafeCommute AI.

Why silence matters: without explicit silent/quiet training data, the model
has no examples of "no sound = safe" and defaults to predicting unsafe for
low-energy input (measured at safe_prob = 0.28 without these samples). In
production, microphones frequently capture near-silence (empty platforms,
quiet bar hours), so the model MUST handle this correctly.

The energy gating in inference.py (RMS < 0.003 = auto-safe) provides a
runtime safety net, but the model should also learn the correct behavior
intrinsically for robustness.

Generates 1,500 synthetic safe-class samples across three RMS tiers:
  - 500 pure silence (RMS = 0.0): digital silence, all zeros. Tests the
    extreme case of no microphone input.
  - 500 very quiet noise (RMS 0.0001-0.001): sensor noise floor. Real
    microphones never produce true silence — there is always thermal noise.
    This range simulates that baseline.
  - 500 low ambient noise (RMS 0.001-0.005): quiet room ambience. Just above
    the energy gate threshold, so the model (not the gate) must classify it.
    The upper bound (0.005) is chosen to slightly exceed ENERGY_GATE_RMS=0.003
    to ensure smooth behavior near the gating boundary.

Split is deterministic 70/15/15 by index (not sha256) because these are
synthetic samples with no source-file identity to hash.

Usage:
    PYTHONPATH=. python safecommute/pipeline/generate_silence_samples.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, TARGET_LENGTH, DATA_DIR
from safecommute.features import extract_features


def generate_samples():
    print("=" * 60)
    print(" Generating synthetic silence/quiet samples")
    print("=" * 60)

    configs = [
        ('synth_silence', 500, 0.0, 0.0),       # pure silence
        ('synth_quiet',   500, 0.0001, 0.001),   # very quiet
        ('synth_ambient', 500, 0.001, 0.005),    # low ambient
    ]

    total = 0
    for prefix, count, rms_lo, rms_hi in configs:
        print(f"\n  {prefix}: {count} samples (RMS {rms_lo}-{rms_hi})")

        for i in range(count):
            # Deterministic split: 70/15/15
            if i < int(count * 0.70):
                split = 'train'
            elif i < int(count * 0.85):
                split = 'val'
            else:
                split = 'test'

            out_dir = os.path.join(DATA_DIR, split, '0_safe')
            os.makedirs(out_dir, exist_ok=True)

            if rms_hi == 0.0:
                # Pure silence
                y = np.zeros(TARGET_LENGTH, dtype=np.float32)
            else:
                # Random noise scaled to target RMS
                target_rms = np.random.uniform(rms_lo, rms_hi)
                y = np.random.randn(TARGET_LENGTH).astype(np.float32)
                current_rms = np.sqrt(np.mean(y ** 2))
                if current_rms > 0:
                    y = y * (target_rms / current_rms)

            features = extract_features(y)
            out_path = os.path.join(out_dir, f"{prefix}_{i:04d}.pt")
            torch.save(features, out_path)
            total += 1

        print(f"    Done: {count} samples")

    print(f"\n  Total generated: {total}")
    print("  All saved as safe class (label=0)")


if __name__ == "__main__":
    generate_samples()
