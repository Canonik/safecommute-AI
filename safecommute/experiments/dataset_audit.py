"""
Deep Dataset Audit for SafeCommute AI.
Analyze every sample, flag noisy/mislabeled data, create quality metrics.
Goal: make our dataset a strong point — clean, well-curated, high-quality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import DATA_DIR, MODEL_SAVE_PATH
from safecommute.dataset import TensorAudioDataset
from research.experiments.eval_utils import load_stats


def analyze_spectrograms(dataset, split_name):
    """Analyze spectrogram quality metrics for all samples."""
    stats = defaultdict(lambda: {'count': 0, 'energy_mean': 0, 'energy_std': 0,
                                  'silence_ratio': 0, 'flat_ratio': 0})
    sample_issues = []

    for i in range(len(dataset)):
        features, label = dataset[i]
        fname = os.path.basename(dataset.filepaths[i])
        source = fname.split('_')[0]

        # Denormalize for analysis
        spec = features.squeeze().numpy()

        # Basic stats
        energy = np.mean(spec ** 2)
        silence_ratio = np.mean(np.abs(spec) < 0.01)  # near-zero regions
        flat_ratio = np.std(spec, axis=1).mean()  # spectral flatness proxy

        # Detect potential issues
        issues = []
        if silence_ratio > 0.8:
            issues.append("mostly_silent")
        if energy < 0.001:
            issues.append("very_low_energy")
        if np.std(spec) < 0.01:
            issues.append("flat_spectrum")
        if np.max(np.abs(spec)) > 10:
            issues.append("extreme_values")
        if np.isnan(spec).any():
            issues.append("contains_nan")

        # Check for duplicate-like patterns (very uniform spectrograms)
        row_var = np.var(spec, axis=1)
        if np.mean(row_var) < 0.001:
            issues.append("uniform_rows")

        stats[source]['count'] += 1
        stats[source]['energy_mean'] += energy
        stats[source]['silence_ratio'] += silence_ratio

        class_name = "safe" if label == 0 else "unsafe"
        if issues:
            sample_issues.append({
                'file': fname,
                'source': source,
                'class': class_name,
                'label': label,
                'issues': issues,
                'energy': float(energy),
                'silence_ratio': float(silence_ratio),
                'index': i,
            })

    # Average stats per source
    for src in stats:
        n = stats[src]['count']
        stats[src]['energy_mean'] /= n
        stats[src]['silence_ratio'] /= n

    return dict(stats), sample_issues


def find_mislabeled_candidates(model, dataset, device, split_name):
    """Use model confidence to flag potentially mislabeled samples.
    Samples where the model is very confident in the WRONG label are suspicious."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mislabeled_candidates = []
    idx = 0

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probs = F.softmax(logits, dim=1)

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = probs[i].argmax().item()
                confidence = probs[i, pred_label].item()
                true_conf = probs[i, true_label].item()

                # Flag: model very confident in wrong prediction
                if pred_label != true_label and confidence > 0.9:
                    fname = os.path.basename(dataset.filepaths[idx])
                    source = fname.split('_')[0]
                    class_name = "safe" if true_label == 0 else "unsafe"
                    pred_name = "safe" if pred_label == 0 else "unsafe"
                    mislabeled_candidates.append({
                        'file': fname,
                        'source': source,
                        'true_label': class_name,
                        'predicted': pred_name,
                        'confidence': float(confidence),
                        'true_confidence': float(true_conf),
                        'index': idx,
                    })
                idx += 1

    return mislabeled_candidates


def analyze_class_distribution(dataset, split_name):
    """Analyze class balance per source."""
    source_class = defaultdict(lambda: defaultdict(int))
    for i, path in enumerate(dataset.filepaths):
        fname = os.path.basename(path)
        source = fname.split('_')[0]
        label = dataset.labels[i]
        class_name = "safe" if label == 0 else "unsafe"
        source_class[source][class_name] += 1

    return dict(source_class)


def generate_report(all_results, save_path="research/dataset_audit_report.md"):
    """Generate comprehensive audit report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write(f"# Dataset Audit Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for split_name, results in all_results.items():
            f.write(f"\n## {split_name.upper()} Split\n\n")

            # Class distribution
            f.write("### Class Distribution by Source\n\n")
            f.write("| Source | Safe | Unsafe | Total | Balance |\n")
            f.write("|--------|------|--------|-------|---------|\n")
            dist = results['distribution']
            for src in sorted(dist.keys()):
                safe = dist[src].get('safe', 0)
                unsafe = dist[src].get('unsafe', 0)
                total = safe + unsafe
                balance = f"{safe/(total)*100:.0f}/{unsafe/(total)*100:.0f}" if total > 0 else "N/A"
                f.write(f"| {src} | {safe} | {unsafe} | {total} | {balance} |\n")

            # Total
            total_safe = sum(d.get('safe', 0) for d in dist.values())
            total_unsafe = sum(d.get('unsafe', 0) for d in dist.values())
            total = total_safe + total_unsafe
            f.write(f"| **TOTAL** | **{total_safe}** | **{total_unsafe}** | **{total}** | "
                    f"**{total_safe/total*100:.0f}/{total_unsafe/total*100:.0f}** |\n\n")

            # Quality issues
            issues = results['issues']
            f.write(f"### Quality Issues ({len(issues)} flagged)\n\n")
            if issues:
                # Summarize by issue type
                issue_counts = defaultdict(int)
                issue_by_source = defaultdict(lambda: defaultdict(int))
                for item in issues:
                    for iss in item['issues']:
                        issue_counts[iss] += 1
                        issue_by_source[item['source']][iss] += 1

                f.write("| Issue | Count | Sources |\n")
                f.write("|-------|-------|---------|\n")
                for iss, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                    sources = ', '.join(f"{s}({c})" for s, c in
                                       sorted(issue_by_source.items()) if iss in issue_by_source[s])
                    f.write(f"| {iss} | {count} | {sources} |\n")

                f.write(f"\n#### Detailed Issues (top 50)\n\n")
                for item in sorted(issues, key=lambda x: -x.get('silence_ratio', 0))[:50]:
                    f.write(f"- **{item['file']}** ({item['class']}): "
                            f"{', '.join(item['issues'])} "
                            f"[energy={item['energy']:.4f}, silence={item['silence_ratio']:.2f}]\n")
            else:
                f.write("No issues detected.\n")

            # Mislabeled candidates
            if 'mislabeled' in results:
                mislabeled = results['mislabeled']
                f.write(f"\n### Potential Mislabeled Samples ({len(mislabeled)} candidates)\n\n")
                if mislabeled:
                    f.write("These samples have >90% model confidence in the WRONG label:\n\n")
                    f.write("| File | Source | True Label | Predicted | Confidence |\n")
                    f.write("|------|--------|------------|-----------|------------|\n")
                    for item in sorted(mislabeled, key=lambda x: -x['confidence'])[:100]:
                        f.write(f"| {item['file']} | {item['source']} | "
                                f"{item['true_label']} | {item['predicted']} | "
                                f"{item['confidence']:.3f} |\n")

                    # Summary by source
                    source_counts = defaultdict(int)
                    for item in mislabeled:
                        source_counts[item['source']] += 1
                    f.write(f"\n**By source:** ")
                    f.write(", ".join(f"{s}: {c}" for s, c in sorted(source_counts.items(), key=lambda x: -x[1])))
                    f.write("\n")
                else:
                    f.write("No high-confidence mislabeled candidates detected.\n")

        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("Based on the audit:\n\n")

        # Count total issues
        all_issues = sum(len(r['issues']) for r in all_results.values())
        all_mislabeled = sum(len(r.get('mislabeled', [])) for r in all_results.values())

        f.write(f"1. **{all_issues} samples** with quality issues (silence, low energy, etc.)\n")
        f.write(f"2. **{all_mislabeled} samples** potentially mislabeled (high-confidence wrong predictions)\n")
        f.write(f"3. Review mislabeled candidates from acted speech sources (CREMA-D, RAVDESS) first\n")
        f.write(f"4. Consider removing mostly-silent samples as they add no signal\n")
        f.write(f"5. Check SAVEE samples carefully — small source with low accuracy\n\n")

    print(f"Report saved: {save_path}")
    return save_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    # Load model for mislabel detection
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))

    all_results = {}

    for split in ['train', 'val', 'test']:
        print(f"\n=== Auditing {split} split ===")
        dataset = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std)
        print(f"  {len(dataset)} samples")

        # Distribution analysis
        print("  Analyzing class distribution...")
        dist = analyze_class_distribution(dataset, split)

        # Spectrogram quality
        print("  Analyzing spectrogram quality...")
        stats, issues = analyze_spectrograms(dataset, split)
        print(f"  Found {len(issues)} samples with quality issues")

        # Mislabel detection
        print("  Detecting potential mislabels...")
        mislabeled = find_mislabeled_candidates(model, dataset, device, split)
        print(f"  Found {len(mislabeled)} potential mislabeled candidates")

        all_results[split] = {
            'distribution': dist,
            'stats': stats,
            'issues': issues,
            'mislabeled': mislabeled,
        }

    # Save detailed results as JSON
    json_path = "research/dataset_audit_details.json"
    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    json_results = {}
    for split, data in all_results.items():
        json_results[split] = make_serializable({
            'distribution': {k: dict(v) for k, v in data['distribution'].items()},
            'issue_count': len(data['issues']),
            'mislabeled_count': len(data.get('mislabeled', [])),
        })

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nDetailed results saved: {json_path}")

    # Generate markdown report
    report_path = generate_report(all_results)
    return report_path


if __name__ == "__main__":
    main()
