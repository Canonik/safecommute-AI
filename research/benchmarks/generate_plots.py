"""
Generate comprehensive benchmark plots from experiment results.
Reads research/experiment_log.md and generates all plots.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


def parse_experiment_log(log_path="research/experiment_log.md"):
    """Parse the experiment log markdown into structured data."""
    with open(log_path) as f:
        content = f.read()

    experiments = []
    # Parse the main table
    lines = content.split('\n')
    for line in lines:
        if line.startswith('|') and not line.startswith('| #') and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 8:
                try:
                    exp = {
                        'num': int(parts[0]),
                        'name': parts[1],
                        'auc': float(parts[2]),
                        'accuracy': float(parts[3]),
                        'f1': float(parts[4]),
                        'params': int(parts[5].replace(',', '')),
                        'size_mb': float(parts[6]),
                        'latency': parts[7],
                        'notes': parts[8] if len(parts) > 8 else '',
                    }
                    # Parse latency
                    lat_match = re.match(r'([\d.]+)±([\d.]+)', exp['latency'])
                    if lat_match:
                        exp['lat_mean'] = float(lat_match.group(1))
                        exp['lat_std'] = float(lat_match.group(2))
                    else:
                        exp['lat_mean'] = 0
                        exp['lat_std'] = 0
                    experiments.append(exp)
                except (ValueError, IndexError):
                    continue

    # Parse per-source breakdowns
    source_data = {}
    sections = content.split('### ')
    for section in sections[1:]:
        lines = section.strip().split('\n')
        exp_name = lines[0].split(' — ')[0].strip()
        source_data[exp_name] = {}
        for line in lines:
            if line.startswith('| ') and not line.startswith('| Source') and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2:
                    try:
                        source_data[exp_name][parts[0]] = float(parts[1])
                    except ValueError:
                        pass

    return experiments, source_data


def plot_auc_comparison(experiments, save_dir):
    """Bar chart: AUC of all experiments vs baseline."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [e['name'] for e in experiments]
    aucs = [e['auc'] for e in experiments]
    baseline_auc = experiments[0]['auc'] if experiments else 0.95

    colors = []
    for auc in aucs:
        if auc > baseline_auc + 0.005:
            colors.append('#2ecc71')  # green - better
        elif auc < baseline_auc - 0.005:
            colors.append('#e74c3c')  # red - worse
        else:
            colors.append('#3498db')  # blue - same

    bars = ax.bar(range(len(names)), aucs, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=baseline_auc, color='orange', linestyle='--', linewidth=2, label=f'Baseline AUC={baseline_auc:.4f}')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('SafeCommute AI — Experiment AUC Comparison')
    ax.set_ylim(min(aucs) - 0.02, max(aucs) + 0.02)
    ax.legend()

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auc_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: auc_comparison.png")


def plot_latency_vs_auc(experiments, save_dir):
    """Scatter: latency vs AUC for all variants."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for exp in experiments:
        ax.scatter(exp['lat_mean'], exp['auc'], s=exp['params']/5000,
                  alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.annotate(exp['name'], (exp['lat_mean'], exp['auc']),
                   fontsize=6, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('SafeCommute AI — Latency vs AUC\n(bubble size = parameter count)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latency_vs_auc.png'), dpi=150)
    plt.close()
    print("  Saved: latency_vs_auc.png")


def plot_radar_comparison(experiments, save_dir):
    """Radar chart: multi-dimensional comparison of top 5 models."""
    # Select top 5 by AUC
    top5 = sorted(experiments, key=lambda e: e['auc'], reverse=True)[:5]

    categories = ['AUC', 'Accuracy', 'F1', 'Speed\n(inv. latency)', 'Efficiency\n(inv. params)']
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for idx, exp in enumerate(top5):
        # Normalize to 0-1 range
        max_lat = max(e['lat_mean'] for e in experiments) if experiments else 1
        max_params = max(e['params'] for e in experiments) if experiments else 1

        values = [
            exp['auc'],
            exp['accuracy'],
            exp['f1'],
            1 - (exp['lat_mean'] / max_lat) if max_lat > 0 else 0,
            1 - (exp['params'] / max_params) if max_params > 0 else 0,
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=exp['name'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('SafeCommute AI — Top 5 Models Radar', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: radar_comparison.png")


def plot_source_heatmap(experiments, source_data, save_dir):
    """Per-source heatmap: accuracy by source × model variant."""
    # Get all sources
    all_sources = set()
    for data in source_data.values():
        all_sources.update(data.keys())
    sources = sorted(all_sources)

    if not sources or not source_data:
        print("  Skipped: source_heatmap.png (no data)")
        return

    # Build matrix
    exp_names = list(source_data.keys())
    matrix = np.zeros((len(exp_names), len(sources)))

    for i, name in enumerate(exp_names):
        for j, src in enumerate(sources):
            matrix[i, j] = source_data[name].get(src, 0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(exp_names) * 0.5)))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, fontsize=9)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=7)

    # Add text annotations
    for i in range(len(exp_names)):
        for j in range(len(sources)):
            text = f'{matrix[i, j]:.2f}'
            color = 'white' if matrix[i, j] < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, label='Accuracy')
    ax.set_title('SafeCommute AI — Per-Source Accuracy Heatmap')
    ax.set_xlabel('Data Source')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'source_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved: source_heatmap.png")


def plot_params_vs_auc(experiments, save_dir):
    """Scatter: parameter count vs AUC."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in experiments:
        color = '#2ecc71' if exp['auc'] >= 0.95 else '#e74c3c'
        ax.scatter(exp['params'] / 1e6, exp['auc'], s=100, c=color, edgecolors='black', linewidth=0.5)
        ax.annotate(exp['name'], (exp['params'] / 1e6, exp['auc']),
                   fontsize=6, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('SafeCommute AI — Parameter Efficiency')
    ax.grid(True, alpha=0.3)

    green_patch = mpatches.Patch(color='#2ecc71', label='AUC ≥ 0.95')
    red_patch = mpatches.Patch(color='#e74c3c', label='AUC < 0.95')
    ax.legend(handles=[green_patch, red_patch])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'params_vs_auc.png'), dpi=150)
    plt.close()
    print("  Saved: params_vs_auc.png")


def generate_comparison_table(experiments, save_path):
    """Generate markdown comparison table."""
    with open(save_path, 'w') as f:
        f.write("# Experiment Comparison Table\n\n")
        f.write("| Rank | Experiment | AUC | Accuracy | F1 | Params | Size(MB) | Latency(ms) | vs Baseline |\n")
        f.write("|------|-----------|-----|----------|----|---------|---------|-----------:|-----------|\n")

        baseline_auc = experiments[0]['auc'] if experiments else 0.95
        sorted_exps = sorted(experiments, key=lambda e: e['auc'], reverse=True)

        for rank, exp in enumerate(sorted_exps, 1):
            delta = exp['auc'] - baseline_auc
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            f.write(f"| {rank} | {exp['name']} | {exp['auc']:.4f} | {exp['accuracy']:.4f} | "
                    f"{exp['f1']:.4f} | {exp['params']:,} | {exp['size_mb']:.2f} | "
                    f"{exp['latency']} | {delta_str} |\n")

        f.write(f"\nBaseline AUC: {baseline_auc:.4f}\n")
        f.write(f"Total experiments: {len(experiments)}\n")

    print(f"  Saved: {save_path}")


def main():
    os.makedirs("research/benchmarks/plots", exist_ok=True)
    save_dir = "research/benchmarks/plots"

    print("=== Generating Benchmark Plots ===")

    experiments, source_data = parse_experiment_log()
    print(f"  Found {len(experiments)} experiments, {len(source_data)} with source breakdowns")

    if not experiments:
        print("  No experiments found in log!")
        return

    plot_auc_comparison(experiments, save_dir)
    plot_latency_vs_auc(experiments, save_dir)
    plot_radar_comparison(experiments, save_dir)
    plot_source_heatmap(experiments, source_data, save_dir)
    plot_params_vs_auc(experiments, save_dir)
    generate_comparison_table(experiments, "research/benchmarks/comparison_table.md")

    print("\n  All plots generated!")


if __name__ == "__main__":
    main()
