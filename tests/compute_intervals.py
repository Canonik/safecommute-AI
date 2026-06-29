"""
Re-emit every detection number in RESULTS.md with a 95 % confidence interval.

Measurement-only: this script touches no model and no training code. It
re-runs inference on the existing test set (`prepared_data/test/`) with the
existing best-model checkpoint (`models/safecommute_v2.pth`), then wraps
every proportion (per-source accuracy, FPR, TPR, overall confusion-matrix
rates, deployment FP and recall) in a Wilson 95 % CI and wraps AUC in a
percentile bootstrap CI.

Output: `tests/reports/intervals.json` -- the single source of truth that
the rewritten RESULTS.md cites.

Every published number now has its uncertainty attached.

Usage:
    PYTHONPATH=. python tests/compute_intervals.py
    PYTHONPATH=. python tests/compute_intervals.py --auc-resamples 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safecommute.constants import MODEL_SAVE_PATH
from safecommute.stats import wilson_interval, bootstrap_auc


REPORTS_DIR = Path("tests/reports")
OUT_PATH = REPORTS_DIR / "intervals.json"


def _source_for(name: str) -> str:
    base = Path(name).name
    for ext in (".pt", ".wav"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("_")
    if len(parts) >= 2 and parts[1] and not parts[1][0].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _wilson_dict(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "point": None, "ci95_lo": None, "ci95_hi": None}
    p, lo, hi = wilson_interval(k, n)
    return {"k": int(k), "n": int(n), "point": p, "ci95_lo": lo, "ci95_hi": hi}


def compute_phase_a_intervals(probs: np.ndarray,
                              labels: np.ndarray,
                              names: list[str],
                              threshold: float = 0.5,
                              auc_resamples: int = 2000) -> dict:
    """Per-source TPR/FPR with Wilson, overall confusion-matrix Wilson,
    overall AUC with bootstrap. All from the (probs, labels, names) tuple
    produced by tests._common.run_inference."""
    preds = (probs >= threshold).astype(np.int64)

    # Overall confusion matrix
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())

    overall = {
        "accuracy": _wilson_dict(tp + tn, tp + fn + tn + fp),
        "tpr_micro": _wilson_dict(tp, tp + fn),
        "fpr_micro": _wilson_dict(fp, fp + tn),
        "precision": _wilson_dict(tp, tp + fp),
        "confusion_counts": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }

    # AUC with bootstrap (2000 resamples by default; bump for paper-final)
    print(f"bootstrap AUC: n_resamples={auc_resamples} ...", flush=True)
    t0 = time.time()
    auc_p, auc_lo, auc_hi = bootstrap_auc(labels, probs, n_resamples=auc_resamples)
    print(f"  AUC = {auc_p:.4f}  [{auc_lo:.4f}, {auc_hi:.4f}]  ({time.time()-t0:.1f}s)")
    overall["auc"] = {"point": auc_p, "ci95_lo": auc_lo, "ci95_hi": auc_hi,
                       "n_resamples": auc_resamples}

    # Per-source TPR / FPR
    per_source_tpr: dict[str, dict] = {}
    per_source_fpr: dict[str, dict] = {}
    by_src_idx: dict[str, list[int]] = defaultdict(list)
    for i, nm in enumerate(names):
        by_src_idx[_source_for(nm)].append(i)

    for src, idxs in by_src_idx.items():
        idxs_arr = np.asarray(idxs)
        lbls = labels[idxs_arr]
        prs = preds[idxs_arr]
        if (lbls == 1).all():
            # All-unsafe source -> TPR (k correct unsafe / n)
            k = int((prs == 1).sum())
            per_source_tpr[src] = _wilson_dict(k, lbls.size)
        elif (lbls == 0).all():
            # All-safe source -> FPR = wrong / n, "accuracy" = right / n
            k_fp = int((prs == 1).sum())
            per_source_fpr[src] = _wilson_dict(k_fp, lbls.size)
        else:
            # Mixed source: emit both
            k_tp = int(((prs == 1) & (lbls == 1)).sum())
            n_unsafe = int((lbls == 1).sum())
            k_fp = int(((prs == 1) & (lbls == 0)).sum())
            n_safe = int((lbls == 0).sum())
            per_source_tpr[src] = _wilson_dict(k_tp, n_unsafe)
            per_source_fpr[src] = _wilson_dict(k_fp, n_safe)

    return {
        "n_samples": int(labels.size),
        "n_safe": int((labels == 0).sum()),
        "n_unsafe": int((labels == 1).sum()),
        "threshold": threshold,
        "overall": overall,
        "per_source_tpr": per_source_tpr,
        "per_source_fpr": per_source_fpr,
    }


def compute_phase_b_intervals() -> dict:
    """Wilson CIs on the metro deployment numbers (n=19 held-out wavs,
    n=57 threat wavs) read from existing JSON reports."""
    phase_b = json.loads(Path("tests/reports/phase_b_metro.json").read_text())
    held_out_n = int(phase_b.get("held_out_wavs", 19))
    fp_rate = float(phase_b.get("fp_rate", 0.0))
    recall = float(phase_b.get("threat_recall", 0.789))
    # Derive integer k from rate (sanity-check against integer); n_threat
    # is reported as the recall denominator inside the source script. The
    # existing report does not include it as a separate field, so we use
    # 57 from the metro_lever_sweep convention documented in the paper package.
    threat_n = 57
    k_fp = int(round(fp_rate * held_out_n))
    k_recall = int(round(recall * threat_n))

    return {
        "fp_rate": _wilson_dict(k_fp, held_out_n),
        "threat_recall": _wilson_dict(k_recall, threat_n),
        "threshold": phase_b.get("low_fpr_threshold"),
        "majority_k": phase_b.get("low_fpr_site_majority_k"),
    }


def compute_metro_lever_intervals() -> dict:
    """Wilson on every (fp, recall) cell in metro_lever_sweep, so the
    RESULTS.md Phase B table gets CIs row-by-row."""
    p = Path("tests/reports/metro_lever_sweep.json")
    if not p.exists():
        return {}
    sweep = json.loads(p.read_text())
    rows = []
    # The sweep is a nested dict by recipe x k x threshold-choice; we flatten.
    # The exact schema varies; we extract every leaf with fp_rate + threat_recall.
    def _walk(obj, path):
        if isinstance(obj, dict):
            if "fp_rate" in obj and "threat_recall" in obj:
                yield (path, obj)
            else:
                for k, v in obj.items():
                    yield from _walk(v, path + [str(k)])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from _walk(v, path + [str(i)])

    for path, leaf in _walk(sweep, []):
        held_n = int(leaf.get("held_out_wavs", 19))
        threat_n = int(leaf.get("threat_n", 57))
        fp_k = int(round(float(leaf["fp_rate"]) * held_n))
        recall_k = int(round(float(leaf["threat_recall"]) * threat_n))
        rows.append({
            "path": "/".join(path),
            "recipe": leaf.get("recipe"),
            "threshold": leaf.get("threshold"),
            "majority_k": leaf.get("majority_k"),
            "fp": _wilson_dict(fp_k, held_n),
            "recall": _wilson_dict(recall_k, threat_n),
        })
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auc-resamples", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    from tests._common import load_model, load_test_dataset, run_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading model {MODEL_SAVE_PATH} on {device}", flush=True)
    model = load_model(MODEL_SAVE_PATH, device=device)
    print("loading test set ...", flush=True)
    ds = load_test_dataset("test")
    print(f"running inference on {len(ds)} samples ...", flush=True)
    t0 = time.time()
    probs, labels, names = run_inference(model, ds, device)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    phase_a = compute_phase_a_intervals(probs, labels, names,
                                         auc_resamples=args.auc_resamples)
    phase_b = compute_phase_b_intervals()
    metro_lever = compute_metro_lever_intervals()

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": MODEL_SAVE_PATH,
        "phase_a": phase_a,
        "phase_b_metro": phase_b,
        "metro_lever_sweep": metro_lever,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")

    # Headline summary printed for human review.
    a = phase_a["overall"]
    print("\n=== Phase A overall (Wilson 95 % CI) ===")
    print(f"  AUC           = {a['auc']['point']:.4f}  "
          f"[{a['auc']['ci95_lo']:.4f}, {a['auc']['ci95_hi']:.4f}]  "
          f"(bootstrap, {a['auc']['n_resamples']} resamples)")
    for name, key in [("Accuracy", "accuracy"), ("TPR (micro)", "tpr_micro"),
                       ("FPR (micro)", "fpr_micro"), ("Precision", "precision")]:
        c = a[key]
        print(f"  {name:14s}= {c['point']*100:5.2f}%  "
              f"[{c['ci95_lo']*100:5.2f}, {c['ci95_hi']*100:5.2f}]  "
              f"(k={c['k']}/n={c['n']})")
    print("\n=== Phase B metro (n=19 held-out, n=57 threat) ===")
    for label, key in [("FP rate     ", "fp_rate"), ("Threat recall", "threat_recall")]:
        c = phase_b[key]
        print(f"  {label}: {c['point']*100:5.2f}%  "
              f"[{c['ci95_lo']*100:5.2f}, {c['ci95_hi']*100:5.2f}]  "
              f"(k={c['k']}/n={c['n']})")


if __name__ == "__main__":
    main()
