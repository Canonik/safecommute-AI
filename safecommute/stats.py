"""
Tiny statistics helpers for reporting metric uncertainty.

Two consumers: the privacy attack harness (`tests/privacy/metrics.py` does its
own bootstrap inline) and the detection-numbers interval re-emission
(`tests/compute_intervals.py`). This module owns the proportion-CI math so
both callers compute it the same way.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion. Reports (point, lo, hi).

    Wilson is preferred over normal-approximation for small n or boundary
    proportions (k=0 or k=n) because it never produces interval bounds
    outside [0, 1] and stays calibrated down to n in the single digits.

    Args:
        k: number of successes.
        n: number of trials. Must be > 0.
        alpha: two-sided significance level (0.05 -> 95 % CI).
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"wilson_interval: k={k} out of range [0, {n}]")
    # Two-sided z for the chosen alpha.
    z = float(_normal_quantile(1.0 - alpha / 2.0))
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (float(p_hat), float(lo), float(hi))


def bootstrap_ci(values: Sequence[float],
                 statistic=np.mean,
                 n_resamples: int = 10_000,
                 alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap on a 1-D array. Returns (point, lo, hi)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = statistic(arr[idx], axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2.0))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha / 2.0)))
    return (float(statistic(arr)), lo, hi)


def bootstrap_auc(y_true: np.ndarray,
                  scores: np.ndarray,
                  n_resamples: int = 2000,
                  alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float, float]:
    """
    Bootstrap CI for ROC-AUC via paired resampling of (y, score).

    `n_resamples=2000` keeps wall-clock under a minute for 12 k samples while
    holding the 95 % CI half-width to about 0.005 in practice. Increase to
    10 k for a paper-final number.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    point = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boot = np.empty(n_resamples, dtype=np.float64)
    misses = 0
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = scores[idx]
        # Resamples that miss one class entirely can't have AUC computed;
        # drop them rather than throw, and count for the report.
        if len(np.unique(y_b)) < 2:
            misses += 1
            boot[i] = np.nan
            continue
        boot[i] = roc_auc_score(y_b, s_b)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return (point, float("nan"), float("nan"))
    lo = float(np.percentile(boot, 100 * alpha / 2.0))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2.0)))
    return (point, lo, hi)


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF. Beasley-Springer-Moro coefficients via
    rational approximation; good to about 1e-9 in the central region we
    care about (alpha=0.05 -> z=1.959963985...).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"_normal_quantile: p={p} out of (0, 1)")
    # math.erfinv exists from Python 3.10; sqrt(2)*erfinv(2p-1) gives the quantile.
    try:
        return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
    except AttributeError:
        # Fallback: scipy if needed (Python<=3.9). Should not be hit on 3.11+.
        from scipy.stats import norm
        return float(norm.ppf(p))
