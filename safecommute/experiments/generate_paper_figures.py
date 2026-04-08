"""
Generate publication-quality figures for SafeCommute AI paper.

Outputs saved to research/figures/ at 300 DPI.

Usage:
    PYTHONPATH=. python research/generate_paper_figures.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, confusion_matrix

from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH
import torch

# Academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

OUTPUT_DIR = 'research/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_analysis_results():
    """Load analysis results from JSON."""
    path = 'analysis/analysis_results.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_model_and_evaluate():
    """Load model and get predictions for all splits."""
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()

    results = {}
    for split in ['train', 'val', 'test']:
        ds = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std)
        probs, labels = [], []
        with torch.no_grad():
            for i in range(len(ds)):
                feat, lab = ds[i]
                logits = model(feat.unsqueeze(0))
                prob = torch.softmax(logits, dim=1)[0][1].item()
                probs.append(prob)
                labels.append(lab.item())
        results[split] = {'probs': np.array(probs), 'labels': np.array(labels)}
    return results


def classify_source(src):
    if src.startswith('as_'):
        return 'AudioSet'
    elif src.startswith('yt_') or src.startswith('viol'):
        return 'Real-world'
    elif src in ('bg', 'hns'):
        return 'UrbanSound8K'
    elif src.startswith('esc'):
        return 'ESC-50'
    return 'Other'


SOURCE_COLORS = {
    'AudioSet': '#e67e22',
    'Real-world': '#27ae60',
    'UrbanSound8K': '#2980b9',
    'ESC-50': '#2980b9',
    'Other': '#7f8c8d',
}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Per-source accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_source_accuracy(analysis):
    per_source = analysis.get('per_source', {})
    if not per_source:
        print("  Skipping per_source_accuracy (no data)")
        return

    sources = sorted(per_source.items(), key=lambda x: x[1]['accuracy'])
    names = [s[0] for s in sources]
    accs = [s[1]['accuracy'] for s in sources]
    totals = [s[1]['total'] for s in sources]
    types = [classify_source(n) for n in names]
    colors = [SOURCE_COLORS.get(t, '#7f8c8d') for t in types]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={t})" for n, t in zip(names, totals)], fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title('Per-Source Test Accuracy')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color='gray', ls='--', alpha=0.4)
    ax.grid(axis='x', alpha=0.2)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=l) for l, c in SOURCE_COLORS.items()
                    if l in types]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_source_accuracy.png'), dpi=300)
    plt.close()
    print("  per_source_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: ROC curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_roc_curves(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    for split, color, ls in [('train', '#2980b9', '-'),
                              ('val', '#e67e22', '--'),
                              ('test', '#27ae60', '-')]:
        fpr, tpr, _ = roc_curve(data[split]['labels'], data[split]['probs'])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(data[split]['labels'], data[split]['probs'])
        ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls,
                label=f'{split.capitalize()} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300)
    plt.close()
    print("  roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig_confusion_matrix(data):
    labels = data['test']['labels']
    preds = (data['test']['probs'] >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Safe', 'Unsafe'])
    ax.set_yticklabels(['Safe', 'Unsafe'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Test Set Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print("  confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Data distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_data_distribution():
    from collections import defaultdict
    source_counts = defaultdict(int)

    for cls in ['0_safe', '1_unsafe']:
        folder = os.path.join(DATA_DIR, 'train', cls)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if not f.endswith('.pt'):
                continue
            parts = f.split('_')
            if parts[0] in ('as', 'yt', 'fsd'):
                src = f"{parts[0]}_{parts[1]}"
            elif parts[0] == 'viol':
                src = 'viol'
            elif parts[0] == 'esc':
                src = 'esc'
            else:
                src = parts[0]
            source_counts[src] += 1

    sources = sorted(source_counts.items(), key=lambda x: -x[1])
    names = [s[0] for s in sources]
    counts = [s[1] for s in sources]
    types = [classify_source(n) for n in names]
    colors = [SOURCE_COLORS.get(t, '#7f8c8d') for t in types]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Training Samples')
    ax.set_title('Training Data Composition by Source')
    ax.grid(axis='x', alpha=0.2)

    for bar, val in zip(bars, counts):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_distribution.png'), dpi=300)
    plt.close()
    print("  data_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Ablation table (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation_table():
    path = 'research/results/ablation_v2_results.json'
    if not os.path.exists(path):
        print("  Skipping ablation_table.tex (no results)")
        return

    with open(path) as f:
        data = json.load(f)

    results = data['results']
    baseline_auc = results[0]['auc']

    tex = r"""\begin{table}[h]
\centering
\caption{Ablation study: contribution of each architectural component.}
\label{tab:ablation}
\begin{tabular}{lccccr}
\toprule
Variant & Params & AUC & Acc & F1 & $\Delta$AUC \\
\midrule
"""
    for r in results:
        delta = r['auc'] - baseline_auc
        sign = '+' if delta >= 0 else ''
        tex += (f"{r['name']} & {r['params']:,} & {r['auc']:.3f} & "
                f"{r['accuracy']:.3f} & {r['f1']:.3f} & {sign}{delta:.3f} \\\\\n")

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(OUTPUT_DIR, 'ablation_table.tex'), 'w') as f:
        f.write(tex)
    print("  ablation_table.tex")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Efficiency table (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────
def fig_efficiency_table():
    path = 'safecommute/benchmark/results/benchmark_results.json'
    if not os.path.exists(path):
        print("  Skipping efficiency_table.tex (no benchmark results)")
        return

    with open(path) as f:
        raw = json.load(f)

    data = raw.get('models', raw)  # handle nested format

    tex = r"""\begin{table}[h]
\centering
\caption{Comparison with SOTA audio classification models on SafeCommute test set.}
\label{tab:efficiency}
\begin{tabular}{lrrrccr}
\toprule
Model & Params & Size (MB) & Latency (ms) & AUC & Acc & F1 \\
\midrule
"""
    for name, r in data.items():
        if not isinstance(r, dict):
            continue
        params = r.get('params', 0)
        params_str = f"{params/1e6:.1f}M" if params > 0 else "0"
        size = r.get('size_mb', 0)
        lat = r.get('latency_mean_ms', 0)
        auc = r.get('auc_roc', 0)
        acc = r.get('accuracy', 0)
        f1 = r.get('f1', 0)
        tex += f"{name} & {params_str} & {size:.0f} & {lat:.0f} & {auc:.3f} & {acc:.3f} & {f1:.3f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(OUTPUT_DIR, 'efficiency_table.tex'), 'w') as f:
        f.write(tex)
    print("  efficiency_table.tex")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: LOSO results
# ─────────────────────────────────────────────────────────────────────────────
def fig_loso_results():
    path = 'research/results/loso_v2_results.json'
    if not os.path.exists(path):
        print("  Skipping loso_results.png (no results)")
        return

    with open(path) as f:
        data = json.load(f)

    results = data['results']
    # Sort by accuracy (since most sources are single-class, AUC is N/A)
    results_sorted = sorted(results, key=lambda x: x['accuracy'])
    names = [r['source'] for r in results_sorted]
    accs = [r['accuracy'] for r in results_sorted]
    types = [classify_source(n) for n in names]
    colors = [SOURCE_COLORS.get(t, '#7f8c8d') for t in types]

    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Accuracy (held-out source)')
    ax.set_title('Leave-One-Source-Out (LOSO) Evaluation')
    ax.axvline(x=np.mean(accs), color='red', ls='--', alpha=0.6,
               label=f'Mean={np.mean(accs):.2f}')
    ax.axvline(x=0.5, color='gray', ls=':', alpha=0.4)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.2)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loso_results.png'), dpi=300)
    plt.close()
    print("  loso_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Generating paper figures")
    print("=" * 60)

    analysis = load_analysis_results()

    print("\n  Loading model and evaluating all splits...")
    data = load_model_and_evaluate()

    print("\n  Generating figures:")
    if analysis:
        fig_per_source_accuracy(analysis)
    fig_roc_curves(data)
    fig_confusion_matrix(data)
    fig_data_distribution()
    fig_ablation_table()
    fig_efficiency_table()
    fig_loso_results()

    print(f"\n  All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
