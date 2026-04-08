"""
Reliability-first evaluation protocol for SafeCommute AI.

Implements:
1) Immutable benchmark freeze/verification
2) Hard-negative stress-suite metrics
3) Reliability gates (hard-negative FPR, threat recalls, worst-source floor)
4) Deployment KPI bundle (alerts/hour, nuisance alert rate, miss rate)
"""
import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, MODEL_SAVE_PATH, STATS_PATH


DEFAULT_HARD_NEG_SOURCES = ["as_laughter", "as_crowd", "as_speech", "yt_metro"]
DEFAULT_THREAT_SOURCES = ["as_screaming", "as_shout", "as_yell"]


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s["mean"], s["std"]


def parse_source_from_path(path):
    fname = os.path.basename(path)
    parts = fname.split("_")
    if len(parts) >= 2 and parts[0] in {"as", "yt"}:
        return f"{parts[0]}_{parts[1]}"
    if parts[0] == "viol":
        return "viol"
    if parts[0] == "esc":
        return "esc"
    return parts[0]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(dataset):
    entries = []
    for p, label in zip(dataset.filepaths, dataset.labels):
        entries.append({
            "path": p,
            "label": int(label),
            "sha256": file_sha256(p),
            "size_bytes": os.path.getsize(p),
            "source": parse_source_from_path(p),
        })
    return {
        "created_at": datetime.now().isoformat(),
        "num_samples": len(entries),
        "entries": entries,
    }


def verify_manifest(dataset, manifest):
    expected = {e["path"]: e for e in manifest.get("entries", [])}
    actual_paths = set(dataset.filepaths)
    expected_paths = set(expected.keys())
    missing = sorted(expected_paths - actual_paths)
    added = sorted(actual_paths - expected_paths)
    changed = []
    for p in sorted(actual_paths & expected_paths):
        e = expected[p]
        if os.path.getsize(p) != e.get("size_bytes") or file_sha256(p) != e.get("sha256"):
            changed.append(p)
    return {"missing": missing, "added": added, "changed": changed}


def predict(model, loader, device):
    probs, preds, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            p_unsafe = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p_unsafe.cpu().tolist())
            preds.extend((p_unsafe >= 0.5).cpu().int().tolist())
            labels.extend(y.tolist())
    return np.array(probs), np.array(preds), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Reliability-first model protocol")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-manifest", type=str,
                        default="research/results/reliability/immutable_benchmark_manifest.json")
    parser.add_argument("--freeze-benchmark", action="store_true")
    parser.add_argument("--hard-neg-sources", type=str, default=",".join(DEFAULT_HARD_NEG_SOURCES))
    parser.add_argument("--threat-sources", type=str, default=",".join(DEFAULT_THREAT_SOURCES))
    parser.add_argument("--max-hard-neg-fpr", type=float, default=0.45)
    parser.add_argument("--min-threat-recall", type=float, default=0.70)
    parser.add_argument("--min-worst-source-acc", type=float, default=0.35)
    parser.add_argument("--windows-per-hour", type=float, default=1200.0)
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--output", type=str,
                        default="research/results/reliability/reliability_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    mean, std = load_stats()
    dataset = TensorAudioDataset(os.path.join(DATA_DIR, args.split), mean, std, augment=False)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples in split: {args.split}")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.benchmark_manifest), exist_ok=True)

    if args.freeze_benchmark:
        with open(args.benchmark_manifest, "w") as f:
            json.dump(build_manifest(dataset), f, indent=2)
        print(f"Frozen immutable benchmark: {args.benchmark_manifest}")

    benchmark_ok = True
    benchmark_drift = {"missing": [], "added": [], "changed": []}
    if os.path.exists(args.benchmark_manifest):
        with open(args.benchmark_manifest) as f:
            manifest = json.load(f)
        benchmark_drift = verify_manifest(dataset, manifest)
        benchmark_ok = not any(benchmark_drift.values())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    probs, _, labels = predict(model, loader, device)
    preds = (probs >= args.threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    hard_neg_sources = [s.strip() for s in args.hard_neg_sources.split(",") if s.strip()]
    threat_sources = [s.strip() for s in args.threat_sources.split(",") if s.strip()]

    source_rows = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fn": 0, "tn": 0, "fp": 0})
    for i, path in enumerate(dataset.filepaths):
        src = parse_source_from_path(path)
        y = int(labels[i])
        p = int(preds[i])
        source_rows[src]["total"] += 1
        source_rows[src]["correct"] += int(y == p)
        if y == 1 and p == 1:
            source_rows[src]["tp"] += 1
        elif y == 1 and p == 0:
            source_rows[src]["fn"] += 1
        elif y == 0 and p == 0:
            source_rows[src]["tn"] += 1
        elif y == 0 and p == 1:
            source_rows[src]["fp"] += 1

    per_source = {}
    for src, row in source_rows.items():
        total = max(1, row["total"])
        pos = row["tp"] + row["fn"]
        neg = row["tn"] + row["fp"]
        per_source[src] = {
            "accuracy": row["correct"] / total,
            "recall": (row["tp"] / pos) if pos > 0 else None,
            "fpr": (row["fp"] / neg) if neg > 0 else None,
            "total": row["total"],
            "tp": row["tp"],
            "fn": row["fn"],
            "tn": row["tn"],
            "fp": row["fp"],
        }

    # Hard-negative stress-suite
    hard_neg_idx = [
        i for i, path in enumerate(dataset.filepaths)
        if int(labels[i]) == 0 and parse_source_from_path(path) in hard_neg_sources
    ]
    hard_neg_fp = int(sum(preds[i] == 1 for i in hard_neg_idx))
    hard_neg_fpr = (hard_neg_fp / len(hard_neg_idx)) if hard_neg_idx else None

    # Threat recalls
    threat_recalls = {}
    for src in threat_sources:
        row = per_source.get(src)
        threat_recalls[src] = None if row is None else row["recall"]

    # Worst-source floor
    source_accuracies = [v["accuracy"] for v in per_source.values() if v["total"] > 0]
    worst_source_acc = min(source_accuracies) if source_accuracies else None

    # Deployment KPI bundle
    safe_idx = [i for i in range(len(labels)) if int(labels[i]) == 0]
    safe_fp_rate = (sum(preds[i] == 1 for i in safe_idx) / len(safe_idx)) if safe_idx else None
    alerts_per_hour_ambient = None if safe_fp_rate is None else safe_fp_rate * args.windows_per_hour
    nuisance_idx = hard_neg_idx
    nuisance_alert_rate = (sum(preds[i] == 1 for i in nuisance_idx) / len(nuisance_idx)) if nuisance_idx else None
    threat_idx = [
        i for i, path in enumerate(dataset.filepaths)
        if int(labels[i]) == 1 and parse_source_from_path(path) in threat_sources
    ]
    threat_miss_rate = (sum(preds[i] == 0 for i in threat_idx) / len(threat_idx)) if threat_idx else None

    gates = {
        "immutable_benchmark": benchmark_ok,
        "hard_negative_fpr": (hard_neg_fpr is not None and hard_neg_fpr <= args.max_hard_neg_fpr),
        "worst_source_floor": (worst_source_acc is not None and worst_source_acc >= args.min_worst_source_acc),
    }
    for src, rec in threat_recalls.items():
        gates[f"threat_recall_{src}"] = (rec is not None and rec >= args.min_threat_recall)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "split": args.split,
        "threshold": args.threshold,
        "benchmark_manifest": args.benchmark_manifest,
        "benchmark_ok": benchmark_ok,
        "benchmark_drift": benchmark_drift,
        "metrics": {
            "auc": None if np.isnan(auc) else float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "hard_negative_fpr": hard_neg_fpr,
            "threat_recalls": threat_recalls,
            "worst_source_accuracy": worst_source_acc,
            "sources": per_source,
        },
        "deployment_kpis": {
            "alerts_per_hour_ambient_only": alerts_per_hour_ambient,
            "nuisance_alert_rate": nuisance_alert_rate,
            "threat_miss_rate": threat_miss_rate,
        },
        "gates": {
            "config": {
                "max_hard_neg_fpr": args.max_hard_neg_fpr,
                "min_threat_recall": args.min_threat_recall,
                "min_worst_source_acc": args.min_worst_source_acc,
            },
            "results": gates,
            "all_pass": all(gates.values()) if gates else False,
        },
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("Reliability protocol summary")
    print("=" * 60)
    print(f"AUC={report['metrics']['auc']}, Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Hard-neg FPR={hard_neg_fpr}, Worst-source acc={worst_source_acc}")
    print(f"Benchmark immutable={benchmark_ok}")
    print(f"All gates pass={report['gates']['all_pass']}")
    print(f"Saved: {args.output}")

    if args.enforce_gates and not report["gates"]["all_pass"]:
        raise SystemExit(1)

    return report


if __name__ == "__main__":
    main()
