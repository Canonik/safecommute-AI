"""
Constrained model selection for reliability-first deployment.

Selects candidates that:
1) Improve shout recall vs baseline
2) Keep crowd and metro accuracy at/above baseline
Then ranks feasible candidates by AUC (or accuracy fallback).
"""
import os
import sys
import glob
import json
import argparse


def load_report(path):
    with open(path) as f:
        return json.load(f)


def source_metric(report, source, field, default=None):
    return report.get("metrics", {}).get("sources", {}).get(source, {}).get(field, default)


def score_value(report):
    auc = report.get("metrics", {}).get("auc")
    if auc is not None:
        return float(auc)
    return float(report.get("metrics", {}).get("accuracy", 0.0))


def main():
    parser = argparse.ArgumentParser(description="Constrained reliability model selection")
    parser.add_argument("--baseline-report", type=str, required=True)
    parser.add_argument("--candidate-glob", type=str, default="research/results/reliability/*.json")
    parser.add_argument("--min-shout-improvement", type=float, default=0.0)
    args = parser.parse_args()

    baseline = load_report(args.baseline_report)
    base_shout = source_metric(baseline, "as_shout", "recall", 0.0)
    base_crowd = source_metric(baseline, "as_crowd", "accuracy", 0.0)
    base_metro = source_metric(baseline, "yt_metro", "accuracy", 0.0)

    candidates = []
    for path in sorted(glob.glob(args.candidate_glob)):
        if os.path.abspath(path) == os.path.abspath(args.baseline_report):
            continue
        try:
            report = load_report(path)
        except Exception:
            continue
        shout = source_metric(report, "as_shout", "recall", 0.0)
        crowd = source_metric(report, "as_crowd", "accuracy", 0.0)
        metro = source_metric(report, "yt_metro", "accuracy", 0.0)
        feasible = (
            # Equality is intentionally allowed for all configured margins.
            shout >= (base_shout + args.min_shout_improvement)
            and crowd >= base_crowd
            and metro >= base_metro
        )
        candidates.append({
            "path": path,
            "score": score_value(report),
            "auc": report.get("metrics", {}).get("auc"),
            "accuracy": report.get("metrics", {}).get("accuracy"),
            "as_shout_recall": shout,
            "as_crowd_accuracy": crowd,
            "yt_metro_accuracy": metro,
            "feasible": feasible,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    feasible = [c for c in candidates if c["feasible"]]

    print("=" * 80)
    print("Constrained model selection")
    print("=" * 80)
    print(f"Baseline: {args.baseline_report}")
    print(f"Baseline as_shout recall: {base_shout:.4f}")
    print(f"Baseline as_crowd accuracy: {base_crowd:.4f}")
    print(f"Baseline yt_metro accuracy: {base_metro:.4f}")
    print("-" * 80)
    for c in candidates:
        tag = "PASS" if c["feasible"] else "FAIL"
        print(
            f"[{tag}] {c['path']} | score={c['score']:.4f} "
            f"| shout={c['as_shout_recall']:.4f} "
            f"| crowd={c['as_crowd_accuracy']:.4f} "
            f"| metro={c['yt_metro_accuracy']:.4f}"
        )

    if feasible:
        best = feasible[0]
        print("-" * 80)
        print(f"Selected: {best['path']}")
        print(f"Selection score: {best['score']:.4f}")
    else:
        print("-" * 80)
        print("No feasible candidate met constraints.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
