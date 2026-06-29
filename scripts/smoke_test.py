#!/usr/bin/env python
"""Lightweight import and inverse-PCEN sanity check for the release snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safecommute.constants as constants
from tests.privacy import inverse_pcen


ORACLE_NMSE_MAX = 1e-8
BLIND_CORR_MIN = 0.80


def main() -> int:
    metrics = inverse_pcen.sanity_check()
    print(json.dumps(metrics, indent=2, default=str))

    failures: list[str] = []
    if metrics["oracle_inverse_nmse_vs_true_mel"] > ORACLE_NMSE_MAX:
        failures.append(
            "oracle inverse NMSE exceeds "
            f"{ORACLE_NMSE_MAX:g}: {metrics['oracle_inverse_nmse_vs_true_mel']:.3g}"
        )
    if metrics["blind_inverse_corr_vs_true_mel"] < BLIND_CORR_MIN:
        failures.append(
            "blind inverse correlation below "
            f"{BLIND_CORR_MIN:g}: {metrics['blind_inverse_corr_vs_true_mel']:.3g}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "smoke test passed: "
        f"{constants.N_MELS} mel bands, {constants.TIME_FRAMES} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
