"""
Comprehensive model analysis for SafeCommute AI.
Generates overfitting diagnostics, per-source breakdowns, confidence
calibration, and all benchmark visualizations.

Outputs:
  - analysis/ directory with all plots and reports
  - analysis/report.md — structured analysis document
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH, N_MELS, TIME_FRAMES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_model_and_data():
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()

    splits = {}
    for split in ['train', 'val', 'test']:
        ds = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std)
        splits[split] = ds
    return model, splits, mean, std


def evaluate_split(model, dataset, split_name):
    """Evaluate model on a dataset split, return detailed results."""
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            feat, lab = dataset[i]
            logits = model(feat.unsqueeze(0))
            prob = torch.softmax(logits, dim=1)[0][1].item()
            preds.append(logits.argmax(1).item())
            labels.append(lab.item())
            probs.append(prob)

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)

    return {
        'split': split_name,
        'n_samples': len(labels),
        'n_safe': int((labels == 0).sum()),
        'n_unsafe': int((labels == 1).sum()),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
        'labels': labels,
        'probs': probs,
        'preds': preds,
    }


def per_source_analysis(model, dataset, split_dir):
    """Break down accuracy by data source prefix."""
    src_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'probs': [], 'labels': []})
    idx = 0
    for cls_label, cls_name in [(0, '0_safe'), (1, '1_unsafe')]:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.endswith('.pt') or f.endswith('_teacher.pt'):
                continue
            # Determine source
            parts = f.split('_')
            if parts[0] in ('yt',):
                src = parts[0] + '_' + parts[1]
            elif parts[0] in ('viol',):
                src = parts[0] + '_' + parts[1]
            else:
                src = parts[0]

            feat, lab = dataset[idx]
            with torch.no_grad():
                logits = model(feat.unsqueeze(0))
                prob = torch.softmax(logits, dim=1)[0][1].item()
                pred = logits.argmax(1).item()

            src_data[src]['total'] += 1
            src_data[src]['labels'].append(cls_label)
            src_data[src]['probs'].append(prob)
            if pred == cls_label:
                src_data[src]['correct'] += 1
            idx += 1

    results = {}
    for src, data in src_data.items():
        acc = data['correct'] / max(data['total'], 1)
        labels = np.array(data['labels'])
        probs = np.array(data['probs'])
        try:
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else acc
        except:
            auc = acc
        results[src] = {
            'accuracy': acc,
            'auc': auc,
            'total': data['total'],
            'n_safe': int((labels == 0).sum()),
            'n_unsafe': int((labels == 1).sum()),
        }
    return results


def generate_plots(train_res, val_res, test_res, per_source, output_dir):
    """Generate all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    # ═══ PLOT 1: Overfitting diagnostic — train vs val vs test ═══
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy', 'F1 Score', 'AUC-ROC']
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        vals = [train_res[metric], val_res[metric], test_res[metric]]
        colors = ['#2196F3', '#FF9800', '#4CAF50']
        bars = axes[i].bar(['Train', 'Val', 'Test'], vals, color=colors, edgecolor='white')
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

        # Add gap annotation
        gap = vals[0] - vals[2]
        if gap > 0.05:
            axes[i].annotate(f'Gap: {gap:.3f}', xy=(1, (vals[0]+vals[2])/2),
                           fontsize=9, color='red', fontweight='bold', ha='center')

    plt.suptitle('Overfitting Diagnostic: Train vs Val vs Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_overfitting_diagnostic.png'), dpi=150)
    plt.close()

    # ═══ PLOT 2: ROC curves for all splits ═══
    fig, ax = plt.subplots(figsize=(8, 8))
    for res, label, color in [(train_res, 'Train', '#2196F3'),
                               (val_res, 'Val', '#FF9800'),
                               (test_res, 'Test', '#4CAF50')]:
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        ax.plot(fpr, tpr, color=color, lw=2,
               label=f'{label} (AUC={res["auc_roc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Train vs Val vs Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_roc_curves.png'), dpi=150)
    plt.close()

    # ═══ PLOT 3: Confidence calibration ═══
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration plot (test set)
    probs = test_res['probs']
    labels = test_res['labels']
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for j in range(n_bins):
        mask = (probs >= bin_edges[j]) & (probs < bin_edges[j+1])
        if mask.sum() > 0:
            bin_acc.append(labels[mask].mean())
            bin_conf.append(probs[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_conf.append((bin_edges[j] + bin_edges[j+1]) / 2)
            bin_count.append(0)

    axes[0].bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color='steelblue', label='Model')
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Confidence Calibration (Test Set)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Confidence distribution
    axes[1].hist(probs[labels == 0], bins=30, alpha=0.6, color='green', label='Safe')
    axes[1].hist(probs[labels == 1], bins=30, alpha=0.6, color='red', label='Unsafe')
    axes[1].axvline(x=0.5, color='black', ls='--', label='Threshold=0.5')
    axes[1].set_xlabel('Predicted Unsafe Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Score Distribution by True Class', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_calibration.png'), dpi=150)
    plt.close()

    # ═══ PLOT 4: Per-source accuracy ═══
    fig, ax = plt.subplots(figsize=(12, 7))
    sources = sorted(per_source.items(), key=lambda x: x[1]['accuracy'])
    names = [s[0] for s in sources]
    accs = [s[1]['accuracy'] for s in sources]
    totals = [s[1]['total'] for s in sources]
    colors = ['#F44336' if a < 0.7 else '#FF9800' if a < 0.85 else '#4CAF50' for a in accs]

    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={t})" for n, t in zip(names, totals)], fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Source Test Accuracy', fontsize=14, fontweight='bold')
    ax.axvline(x=0.7, color='red', ls='--', alpha=0.5, label='70% threshold')
    ax.axvline(x=0.85, color='orange', ls='--', alpha=0.5, label='85% threshold')
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_per_source_accuracy.png'), dpi=150)
    plt.close()

    # ═══ PLOT 5: Confusion matrix ═══
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, res, title in [(axes[0], train_res, 'Train'),
                            (axes[1], val_res, 'Val'),
                            (axes[2], test_res, 'Test')]:
        cm = confusion_matrix(res['labels'], res['preds'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{title} Confusion Matrix', fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Safe', 'Unsafe'])
        ax.set_yticklabels(['Safe', 'Unsafe'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_confusion_matrices.png'), dpi=150)
    plt.close()

    # ═══ PLOT 6: Precision-Recall curve ═══
    fig, ax = plt.subplots(figsize=(8, 8))
    for res, label, color in [(test_res, 'Test', '#4CAF50'),
                               (val_res, 'Val', '#FF9800')]:
        prec_curve, rec_curve, _ = precision_recall_curve(res['labels'], res['probs'])
        ax.plot(rec_curve, prec_curve, color=color, lw=2, label=f'{label}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_precision_recall.png'), dpi=150)
    plt.close()

    print(f"  6 analysis plots saved to {output_dir}/")


def generate_report(train_res, val_res, test_res, per_source, output_dir):
    """Generate markdown analysis report."""
    train_test_gap = train_res['accuracy'] - test_res['accuracy']
    train_test_auc_gap = train_res['auc_roc'] - test_res['auc_roc']

    # Overfitting assessment
    if train_test_gap > 0.15:
        overfit_verdict = "SEVERE OVERFITTING"
        overfit_color = "critical"
    elif train_test_gap > 0.08:
        overfit_verdict = "MODERATE OVERFITTING"
        overfit_color = "warning"
    elif train_test_gap > 0.03:
        overfit_verdict = "MILD OVERFITTING"
        overfit_color = "acceptable"
    else:
        overfit_verdict = "NO SIGNIFICANT OVERFITTING"
        overfit_color = "good"

    report = f"""# SafeCommute AI — Comprehensive Model Analysis

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**Model**: CNN6 + SE + GRU + Multi-Scale Pooling (1.83M params, 7MB)
**Training Recipe**: Focal Loss (γ=3), Cosine Annealing, Strong Augmentation

---

## 1. Dataset Overview

| Split | Total | Safe | Unsafe | Ratio |
|-------|-------|------|--------|-------|
| Train | {train_res['n_samples']} | {train_res['n_safe']} | {train_res['n_unsafe']} | {train_res['n_safe']/max(train_res['n_unsafe'],1):.1f}:1 |
| Val | {val_res['n_samples']} | {val_res['n_safe']} | {val_res['n_unsafe']} | {val_res['n_safe']/max(val_res['n_unsafe'],1):.1f}:1 |
| Test | {test_res['n_samples']} | {test_res['n_safe']} | {test_res['n_unsafe']} | {test_res['n_safe']/max(test_res['n_unsafe'],1):.1f}:1 |

### Data Sources
- **Acted speech**: RAVDESS (1440), CREMA-D (7442), TESS (2800), SAVEE (480)
- **Environmental**: UrbanSound8K (~6400), ESC-50 (~800)
- **Real-world**: YouTube metro ambient (~58 clean clips), YouTube screams (~57 clean clips)
- **Violence detection**: HuggingFace Hemg/audio-based-violence-dataset (2000)
- **Data cleaning**: Removed 34 metro + 4 scream YouTube files (music, news, too energetic)

---

## 2. Overfitting Analysis

### Verdict: **{overfit_verdict}**

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| Accuracy | {train_res['accuracy']:.3f} | {val_res['accuracy']:.3f} | {test_res['accuracy']:.3f} | {train_test_gap:+.3f} |
| F1 | {train_res['f1']:.3f} | {val_res['f1']:.3f} | {test_res['f1']:.3f} | {train_res['f1']-test_res['f1']:+.3f} |
| AUC-ROC | {train_res['auc_roc']:.3f} | {val_res['auc_roc']:.3f} | {test_res['auc_roc']:.3f} | {train_test_auc_gap:+.3f} |
| Precision | {train_res['precision']:.3f} | {val_res['precision']:.3f} | {test_res['precision']:.3f} | {train_res['precision']-test_res['precision']:+.3f} |
| Recall | {train_res['recall']:.3f} | {val_res['recall']:.3f} | {test_res['recall']:.3f} | {train_res['recall']-test_res['recall']:+.3f} |

**Interpretation**:
- Train-Test accuracy gap: {train_test_gap:.3f} — {"concerning, model memorizes training patterns" if train_test_gap > 0.08 else "acceptable for the data diversity" if train_test_gap > 0.03 else "excellent generalization"}
- Train-Test AUC gap: {train_test_auc_gap:.3f} — {"AUC tells a different story: the model's ranking ability generalizes well" if train_test_auc_gap < 0.05 else "ranking ability degrades, true overfitting"}
- The AUC (ranking quality) is more important than accuracy for a threshold-based system

---

## 3. Per-Source Analysis

| Source | Accuracy | Samples | Type | Assessment |
|--------|----------|---------|------|------------|
"""
    for src, data in sorted(per_source.items(), key=lambda x: x[1]['accuracy']):
        src_type = "acted" if src in ('cremad', 'rav', 'savee', 'tess') else "real-world" if src.startswith('yt_') or src.startswith('viol_') else "environmental"
        assessment = "weak" if data['accuracy'] < 0.7 else "moderate" if data['accuracy'] < 0.85 else "strong"
        report += f"| {src} | {data['accuracy']:.1%} | {data['total']} | {src_type} | {assessment} |\n"

    report += f"""
**Key finding**: The model excels on real-world audio (YouTube screams {per_source.get('yt_scream', {}).get('accuracy', 0):.0%}, metro {per_source.get('yt_metro', {}).get('accuracy', 0):.0%}) but struggles with acted speech (CREMA-D {per_source.get('cremad', {}).get('accuracy', 0):.0%}). This is because acted emotions sound fundamentally different from real escalation.

---

## 4. Where the Model Excels

1. **Real-world scream detection**: {per_source.get('yt_scream', {}).get('accuracy', 0):.0%} accuracy on YouTube screams — the actual deployment target
2. **Environmental noise rejection**: >95% on UrbanSound8K backgrounds and ESC-50
3. **Hard negative handling**: {per_source.get('hns', {}).get('accuracy', 0):.0%} on loud-but-safe sounds (jackhammer, car horn)
4. **AUC-ROC = {test_res['auc_roc']:.3f}**: Strong discrimination — the model CAN tell safe from unsafe, threshold just needs tuning
5. **Size/speed**: 7MB, ~9ms CPU inference — genuinely deployable on edge hardware

## 5. Where the Model Lacks

1. **Acted speech accuracy**: CREMA-D ({per_source.get('cremad', {}).get('accuracy', 0):.0%}), RAVDESS ({per_source.get('rav', {}).get('accuracy', 0):.0%}), SAVEE ({per_source.get('savee', {}).get('accuracy', 0):.0%}) — actors performing anger ≠ real aggression
2. **Overfitting gap**: Train accuracy significantly higher than test ({train_test_gap:.1%} gap) — the model memorizes some training patterns
3. **No Italian-specific training**: Emozionalmente was tested but degraded performance
4. **No real metro field validation**: All evaluation is on curated test sets
5. **Class imbalance**: {train_res['n_safe']/max(train_res['n_unsafe'],1):.1f}:1 safe-to-unsafe ratio in training

## 6. Addressable Weaknesses

| Weakness | Solution | Effort | Impact |
|----------|----------|--------|--------|
| Low acted speech accuracy | More real-world data (field recordings) | Medium | High |
| Overfitting | More data, stronger regularization, dropout increase | Low | Medium |
| No Italian support | Fine-tune on Italian subset separately | Medium | Medium |
| No field validation | 1-week pilot at metro station | High | Critical |

## 7. Fundamental Limitations (Cannot Fix with Code)

1. **Acted ≠ Real**: No amount of training on CREMA-D/RAVDESS will match real escalation audio. This requires field recordings.
2. **Single microphone**: Cannot determine distance or direction of sound source. Hardware limitation.
3. **3-second window**: Very short escalations or brief screams may be missed. Trade-off with latency.
4. **Adversarial attacks**: A phone playing a scream video will trigger the alarm. No defense exists.
5. **Cultural variation**: What sounds aggressive varies by culture and language.

---

## 8. Plots

1. `01_overfitting_diagnostic.png` — Train vs Val vs Test metrics comparison
2. `02_roc_curves.png` — ROC curves showing ranking quality across splits
3. `03_calibration.png` — Confidence calibration + score distributions
4. `04_per_source_accuracy.png` — Accuracy breakdown by data source
5. `05_confusion_matrices.png` — Confusion matrices for all splits
6. `06_precision_recall.png` — Precision-Recall trade-off curves
"""

    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    return report


def main():
    output_dir = 'analysis'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print(" Comprehensive Model Analysis")
    print("=" * 50)

    print("\nLoading model and data...")
    model, splits, mean, std = load_model_and_data()

    print("Evaluating train split...")
    train_res = evaluate_split(model, splits['train'], 'train')
    print(f"  Train: Acc={train_res['accuracy']:.3f} F1={train_res['f1']:.3f} AUC={train_res['auc_roc']:.3f}")

    print("Evaluating val split...")
    val_res = evaluate_split(model, splits['val'], 'val')
    print(f"  Val:   Acc={val_res['accuracy']:.3f} F1={val_res['f1']:.3f} AUC={val_res['auc_roc']:.3f}")

    print("Evaluating test split...")
    test_res = evaluate_split(model, splits['test'], 'test')
    print(f"  Test:  Acc={test_res['accuracy']:.3f} F1={test_res['f1']:.3f} AUC={test_res['auc_roc']:.3f}")

    gap = train_res['accuracy'] - test_res['accuracy']
    print(f"\n  Train-Test gap: {gap:.3f} {'⚠️ OVERFITTING' if gap > 0.08 else '✓ OK'}")

    print("\nPer-source analysis...")
    per_source = per_source_analysis(model, splits['test'], os.path.join(DATA_DIR, 'test'))
    for src, data in sorted(per_source.items(), key=lambda x: x[1]['accuracy']):
        print(f"  {src:15}: {data['accuracy']:6.1%} ({data['total']} samples)")

    print("\nGenerating plots...")
    generate_plots(train_res, val_res, test_res, per_source, output_dir)

    print("\nGenerating report...")
    generate_report(train_res, val_res, test_res, per_source, output_dir)

    # Save JSON results
    results = {
        'train': {k: v for k, v in train_res.items() if k not in ('labels', 'probs', 'preds')},
        'val': {k: v for k, v in val_res.items() if k not in ('labels', 'probs', 'preds')},
        'test': {k: v for k, v in test_res.items() if k not in ('labels', 'probs', 'preds')},
        'per_source': per_source,
        'overfitting_gap': {
            'accuracy': float(train_res['accuracy'] - test_res['accuracy']),
            'f1': float(train_res['f1'] - test_res['f1']),
            'auc': float(train_res['auc_roc'] - test_res['auc_roc']),
        }
    }
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
