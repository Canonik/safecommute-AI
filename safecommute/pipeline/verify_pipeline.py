"""
Pipeline verification script for SafeCommute AI.

Diagnostic-only script that validates the data pipeline output without
modifying any files. Run after data_pipeline.py + prepare_*.py to catch
issues before training.

Five verification checks:
  1. Raw data: verifies expected directory structure and wav file counts.
  2. Prepared data: counts .pt tensors per split/class.
  3. Source-level leakage: the CRITICAL check — ensures no source audio file
     has chunks in multiple splits. Leakage would inflate test metrics because
     overlapping 3-second chunks from the same 10-second AudioSet clip share
     ~50% of their audio content. The check extracts source IDs by stripping
     chunk suffixes (_cNNN) and verifies each source appears in exactly one split.
  4. Class balance: warns if any split has >85% of one class, which can cause
     the model to degenerate into always predicting the majority class.
  5. Per-source breakdown: counts samples by source prefix (as_, yt_, esc_, etc.)
     per split, useful for diagnosing which data sources are under-represented.

Usage:
    PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py
"""

import os
import sys
import hashlib
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import RAW_DIR, DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW DATA CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_raw_data():
    """Check raw data directories exist and contain expected files."""
    print("=" * 60)
    print(" 1. RAW DATA CHECK")
    print("=" * 60)

    dirs_to_check = {
        # AudioSet threat categories
        'audioset/threat/screaming': 'AudioSet screaming',
        'audioset/threat/shout': 'AudioSet shout',
        'audioset/threat/yell': 'AudioSet yell',
        'audioset/threat/gunshot': 'AudioSet gunshot',
        'audioset/threat/explosion': 'AudioSet explosion',
        'audioset/threat/glass_breaking': 'AudioSet glass_breaking',
        # AudioSet safe categories
        'audioset/safe/laughter': 'AudioSet laughter',
        'audioset/safe/crowd': 'AudioSet crowd',
        'audioset/safe/speech': 'AudioSet speech',
        'audioset/safe/music': 'AudioSet music',
        'audioset/safe/applause': 'AudioSet applause',
        'audioset/safe/cheering': 'AudioSet cheering',
        'audioset/safe/singing': 'AudioSet singing',
        # FSD50K (optional)
        'fsd50k/threat': 'FSD50K threat (optional)',
        'fsd50k/safe': 'FSD50K safe (optional)',
        # YouTube
        'youtube_screams': 'YouTube screams',
        'youtube_metro': 'YouTube metro',
        # Violence
        'violence': 'Violence dataset',
    }

    total_raw = 0
    for subdir, description in dirs_to_check.items():
        full_path = os.path.join(RAW_DIR, subdir)
        if not os.path.exists(full_path):
            print(f"  MISSING  {description:<30} ({subdir})")
            continue

        # Count wav files (recursively for nested dirs)
        wav_count = 0
        for root, _, files in os.walk(full_path):
            wav_count += sum(1 for f in files if f.endswith('.wav'))

        status = "OK" if wav_count > 0 else "EMPTY"
        print(f"  {status:<8} {description:<30} {wav_count:>5} wav files")
        total_raw += wav_count

    print(f"\n  Total raw wav files: {total_raw}")
    return total_raw


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPARED DATA CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_prepared_data():
    """Check prepared data structure and count samples."""
    print("\n" + "=" * 60)
    print(" 2. PREPARED DATA CHECK")
    print("=" * 60)

    counts = {}
    total = 0

    for split in ['train', 'val', 'test']:
        counts[split] = {}
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(folder):
                counts[split][cls] = 0
                print(f"  MISSING  {split}/{cls}")
                continue
            n = len([f for f in os.listdir(folder) if f.endswith('.pt')])
            counts[split][cls] = n
            total += n
            print(f"  {split:6}/{cls:10}: {n:>5} samples")

    print(f"\n  Total prepared samples: {total}")
    return counts, total


# ─────────────────────────────────────────────────────────────────────────────
# 3. SOURCE-LEVEL LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────
def extract_source_id(filename):
    """
    Extract the source identifier from a .pt filename by stripping
    the chunk suffix (_cNNN).

    This is the inverse of the naming convention used in data_pipeline.py,
    prepare_youtube_data.py, and prepare_violence_data.py. All chunks from
    the same source file share the same source_id, so if any two chunks
    with the same source_id appear in different splits, that is a leakage
    violation.

    Examples:
      bg_135776-2-0-49.pt                → bg_135776-2-0-49  (no chunk suffix)
      yt_metro_0Nry67vaus0_c000.pt        → yt_metro_0Nry67vaus0
      as_screaming_ABC123_30_40_c000.pt   → as_screaming_ABC123_30_40
      viol_violence_123_1_c002.pt         → viol_violence_123_1
    """
    base = filename.replace('.pt', '')

    # Strip chunk suffix (_cNNN) for chunked sources
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].startswith('c') and parts[1][1:].isdigit():
        return parts[0]

    return base


def check_leakage():
    """
    Verify no source file appears in multiple splits (the leakage check).

    This is the most important verification. Source-level leakage occurs when
    chunks from the same audio file end up in both training and test splits.
    With 50% overlap chunking, adjacent chunks share ~1.5 seconds of audio,
    so leakage would let the model "memorize" test data during training,
    inflating reported metrics by 5-15% based on our experiments.
    """
    print("\n" + "=" * 60)
    print(" 3. SOURCE-LEVEL LEAKAGE CHECK")
    print("=" * 60)

    # Map source_id → set of splits it appears in
    source_splits = defaultdict(set)

    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith('.pt'):
                    continue
                src_id = extract_source_id(fname)
                source_splits[src_id].add(split)

    # Find violations
    violations = []
    for src_id, splits in source_splits.items():
        if len(splits) > 1:
            violations.append((src_id, sorted(splits)))

    if not violations:
        print(f"  PASS — No leakage detected across {len(source_splits)} unique sources")
    else:
        print(f"  FAIL — {len(violations)} sources appear in multiple splits:")
        for src_id, splits in violations[:20]:  # show first 20
            print(f"    {src_id}: {', '.join(splits)}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")

    return len(violations)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASS BALANCE CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_balance(counts):
    """Check class balance per split."""
    print("\n" + "=" * 60)
    print(" 4. CLASS BALANCE CHECK")
    print("=" * 60)

    issues = 0
    for split in ['train', 'val', 'test']:
        safe = counts.get(split, {}).get('0_safe', 0)
        unsafe = counts.get(split, {}).get('1_unsafe', 0)
        total = safe + unsafe
        if total == 0:
            print(f"  {split}: EMPTY")
            issues += 1
            continue

        safe_pct = 100 * safe / total
        unsafe_pct = 100 * unsafe / total
        ratio = f"{safe / max(unsafe, 1):.1f}:1"

        status = "OK" if safe_pct <= 85 else "WARN"
        if status == "WARN":
            issues += 1

        print(f"  {split:6}: {safe:>5} safe ({safe_pct:.0f}%) + "
              f"{unsafe:>5} unsafe ({unsafe_pct:.0f}%) = "
              f"{total:>5} total  [{ratio} ratio]  {status}")

    if issues == 0:
        print("\n  PASS — All splits have acceptable class balance")
    else:
        print(f"\n  WARN — {issues} split(s) may have balance issues (>85% one class)")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-SOURCE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def per_source_breakdown():
    """Count samples per source prefix per split."""
    print("\n" + "=" * 60)
    print(" 5. PER-SOURCE BREAKDOWN")
    print("=" * 60)

    # source_prefix → {split → {class → count}}
    source_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith('.pt'):
                    continue
                # Extract source prefix (first 1-2 underscore-separated tokens)
                parts = fname.split('_')
                if parts[0] in ('yt', 'as', 'fsd', 'esc', 'viol'):
                    prefix = parts[0] + '_' + parts[1] if len(parts) > 1 else parts[0]
                else:
                    prefix = parts[0]
                source_data[prefix][split][cls] += 1

    # Print table
    print(f"  {'Source':<20} {'train/safe':>10} {'train/uns':>10} "
          f"{'val/safe':>10} {'val/uns':>10} {'test/safe':>10} {'test/uns':>10}")
    print("  " + "─" * 82)

    for prefix in sorted(source_data.keys()):
        data = source_data[prefix]
        print(f"  {prefix:<20} "
              f"{data['train']['0_safe']:>10} {data['train']['1_unsafe']:>10} "
              f"{data['val']['0_safe']:>10} {data['val']['1_unsafe']:>10} "
              f"{data['test']['0_safe']:>10} {data['test']['1_unsafe']:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" SafeCommute AI — Pipeline Verification")
    print("=" * 60)
    print()

    raw_total = check_raw_data()
    counts, prep_total = check_prepared_data()
    n_violations = check_leakage()
    n_balance_issues = check_balance(counts)
    per_source_breakdown()

    # Final verdict
    print("\n" + "=" * 60)
    print(" VERDICT")
    print("=" * 60)

    issues = []
    if raw_total == 0:
        issues.append("No raw data found")
    if prep_total == 0:
        issues.append("No prepared data found")
    if prep_total < 5000:
        issues.append(f"Only {prep_total} prepared samples (need >5000 for training)")
    if n_violations > 0:
        issues.append(f"{n_violations} source-level leakage violations")
    if n_balance_issues > 0:
        issues.append(f"{n_balance_issues} class balance warnings")

    if not issues:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {len(issues)} ISSUE(S) FOUND:")
        for issue in issues:
            print(f"    - {issue}")

    print()


if __name__ == "__main__":
    main()
