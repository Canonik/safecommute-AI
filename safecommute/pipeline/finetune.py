"""
Fine-tune the base SafeCommute model for a specific deployment environment.

The base model (trained on Layer 1 universal threats + Layer 2 hard negatives)
is adapted to a target environment (metro, bar, bus) by injecting Layer 3
deployment-specific ambient audio into the safe class.

This is the key personalization step: threat sounds are universal (screams,
gunshots, glass breaking sound the same everywhere), but what counts as
"normal background" changes per deployment. A metro station has announcements,
train braking sounds, and crowd noise that a bar does not. Fine-tuning shifts
the safe-class decision boundary to encompass these environment-specific sounds
without retraining the threat-detection capability from scratch.

Catastrophic forgetting prevention strategy:
  - The fine-tuning dataset combines ALL base unsafe samples (Layer 1) with a
    SUBSET of base safe samples (Layer 2, controlled by --keep-safe-ratio)
    plus the new ambient data (Layer 3). Keeping all unsafe samples ensures
    the model does not "forget" what threats sound like.
  - Optional CNN freezing (--freeze-cnn): locks the convolutional feature
    extractor and only trains GRU + FC layers. This preserves learned spectral
    features while adapting temporal/decision layers.
  - Warmup strategy (--warmup-epochs): starts with CNN frozen for N epochs
    (letting GRU/FC adapt to the new data distribution), then unfreezes CNN
    with a 10x lower learning rate. This two-phase approach prevents early
    gradient updates from destroying pretrained CNN filters.
  - Lower learning rate (default 1e-4, vs 3e-4 for base training): smaller
    updates preserve base model knowledge.
  - Focal loss gamma=3.0 (vs 2.0 for base): more aggressive hard-example
    focusing because the fine-tuning set is more imbalanced.

Threshold optimization:
  After fine-tuning, three detection thresholds are computed on the test set:
    - Youden's J: maximizes (sensitivity + specificity - 1), balanced.
    - F1-optimal: maximizes weighted F1-score, biased toward majority class.
    - Low-FPR: highest threshold where FPR <= 5%, conservative for deployment
      (minimizes false alarms at the cost of some missed detections).
  The deployment typically uses the low-FPR threshold because false alarms
  erode user trust faster than occasional missed detections.

Usage:
    PYTHONPATH=. python safecommute/pipeline/finetune.py \\
        --environment metro --ambient-dir raw_data/youtube_metro --freeze-cnn

    PYTHONPATH=. python safecommute/pipeline/finetune.py \\
        --environment bar --ambient-dir raw_data/recorded_bar --epochs 15
"""

import os
import sys
import json
import shutil
import random
import argparse
import tempfile

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import (
    SAMPLE_RATE, DATA_DIR, STATS_PATH, MODEL_SAVE_PATH,
)
from safecommute.features import extract_features, pad_or_truncate, chunk_audio
from safecommute.utils import seed_everything, worker_init_fn
from safecommute.pipeline.train import FocalLoss, mixup_batch

BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
MIXUP_PROB = 0.3


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def process_ambient_audio(ambient_dir, output_dir):
    """
    Process ambient .wav files into .pt spectrograms in a temp directory.
    All files are treated as safe (label=0).
    Returns count of processed chunks.
    """
    safe_dir = os.path.join(output_dir, '0_safe')
    unsafe_dir = os.path.join(output_dir, '1_unsafe')
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)  # empty, needed for TensorAudioDataset

    count = 0
    for fname in sorted(os.listdir(ambient_dir)):
        if not fname.endswith('.wav'):
            continue
        path = os.path.join(ambient_dir, fname)
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            if len(y) < SAMPLE_RATE:
                continue

            # Chunk long audio into 3-second windows
            chunks = chunk_audio(y)
            base = fname.replace('.wav', '')
            for i, chunk in enumerate(chunks):
                features = extract_features(chunk)
                out_path = os.path.join(safe_dir, f"ambient_{base}_c{i:03d}.pt")
                torch.save(features, out_path)
                count += 1
        except Exception as e:
            print(f"  Skip {fname}: {e}")

    return count


def subsample_dataset(dataset, ratio):
    """Return a Subset with a random fraction of the dataset (unused but kept for API)."""
    n = len(dataset)
    k = max(1, int(n * ratio))
    indices = random.sample(range(n), k)
    return Subset(dataset, indices)


def evaluate_model(model, test_loader, device):
    """Evaluate and return AUC, accuracy, F1."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')
    return auc, acc, f1


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune SafeCommute for a deployment environment')
    parser.add_argument('--base-model', type=str, default=MODEL_SAVE_PATH,
                        help='Base model checkpoint path')
    parser.add_argument('--environment', type=str, required=True,
                        help='Environment name (metro, bar, bus, etc.)')
    parser.add_argument('--ambient-dir', type=str, required=True,
                        help='Directory with ambient .wav files')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Fine-tuning epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--freeze-cnn', action='store_true',
                        help='Freeze CNN layers, only train GRU+FC')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Epochs with CNN frozen before unfreezing (default: 0)')
    parser.add_argument('--keep-safe-ratio', type=float, default=0.5,
                        help='Ratio of base safe data to keep (default: 0.5)')
    parser.add_argument('--calibration-ambient-dir', type=str, default=None,
                        help='Directory of held-out site ambient .wav files '
                             '(never seen during fine-tuning) used ONLY to '
                             'calibrate a site-specific "low_fpr_site" '
                             'threshold. Recommended: a held-out 20%% split '
                             'of the target-site ambient pool. If absent, '
                             'only the combined-test-set low_fpr is emitted.')
    parser.add_argument('--calibration-majority-k', type=int, default=2,
                        help='Temporal-majority aggregation k for the site '
                             'threshold sweep (default: 2). See '
                             'test_deployment.py --majority-k.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    print("=" * 60)
    print(f" SafeCommute AI — Fine-tune for: {args.environment}")
    print("=" * 60)

    # ── Step 1: Load base model ──────────────────────────────────────
    print(f"\n  Loading base model: {args.base_model}")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(
        torch.load(args.base_model, map_location=device, weights_only=True))

    # ── Step 2: Process ambient audio ────────────────────────────────
    print(f"  Processing ambient audio from: {args.ambient_dir}")
    tmp_dir = tempfile.mkdtemp(prefix=f"safecommute_ft_{args.environment}_")
    n_ambient = process_ambient_audio(args.ambient_dir, tmp_dir)
    print(f"  Processed {n_ambient} ambient chunks → safe class")

    if n_ambient == 0:
        print("  ERROR: No ambient audio processed. Check --ambient-dir.")
        shutil.rmtree(tmp_dir)
        return

    # ── Step 3: Build fine-tuning dataset ────────────────────────────
    print("\n  Building fine-tuning dataset...")

    # Ambient data (new safe samples for this environment)
    ambient_dataset = TensorAudioDataset(tmp_dir, mean, std, augment=True)

    # Base training data (to prevent catastrophic forgetting)
    base_train = TensorAudioDataset(
        os.path.join(DATA_DIR, 'train'), mean, std, augment=True)

    # Split base training into safe and unsafe indices
    safe_indices = [i for i, l in enumerate(base_train.labels) if l == 0]
    unsafe_indices = [i for i, l in enumerate(base_train.labels) if l == 1]

    # Keep a subset of base safe data + ALL base unsafe data
    n_keep_safe = max(1, int(len(safe_indices) * args.keep_safe_ratio))
    kept_safe = random.sample(safe_indices, n_keep_safe)
    base_subset = Subset(base_train, kept_safe + unsafe_indices)

    # Combine: ambient safe + subset of base data
    combined_dataset = ConcatDataset([ambient_dataset, base_subset])

    print(f"  Ambient safe chunks: {n_ambient}")
    print(f"  Base safe (kept {args.keep_safe_ratio:.0%}): {n_keep_safe}")
    print(f"  Base unsafe (all): {len(unsafe_indices)}")
    print(f"  Total fine-tuning samples: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)

    # Test set (base test — threat detection must not degrade)
    test_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'test'), mean, std)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    # ── Step 4: Evaluate base model before fine-tuning ───────────────
    base_auc, base_acc, base_f1 = evaluate_model(model, test_loader, device)
    print(f"\n  Base model:  AUC={base_auc:.4f}  Acc={base_acc:.4f}  F1={base_f1:.4f}")

    # ── Step 5: Fine-tune ────────────────────────────────────────────
    # Warmup = start with CNN frozen, then unfreeze after N epochs.
    # This two-phase approach lets the GRU/FC layers adapt to the new
    # data distribution before allowing gradients to flow back into the
    # CNN feature extractor, preventing catastrophic forgetting of
    # learned spectral features.
    use_warmup = args.warmup_epochs > 0 and not args.freeze_cnn
    print(f"\n  Fine-tuning: {args.epochs} epochs, lr={args.lr}, "
          f"freeze_cnn={args.freeze_cnn}, warmup_epochs={args.warmup_epochs}")

    def freeze_cnn_layers():
        """Freeze CNN blocks + input BN, keep GRU/freq_reduce/FC trainable."""
        for name, param in model.named_parameters():
            if 'block' in name or 'bn_input' in name:
                param.requires_grad = False

    def unfreeze_all():
        for param in model.parameters():
            param.requires_grad = True

    if args.freeze_cnn or use_warmup:
        freeze_cnn_layers()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  CNN frozen, training {trainable:,}/{total_params:,} params")
        if use_warmup:
            print(f"  Will unfreeze after {args.warmup_epochs} warmup epochs")

    # Class weights from combined dataset
    all_labels = []
    for i in range(len(combined_dataset)):
        _, label = combined_dataset[i]
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    counts = [all_labels.count(0), all_labels.count(1)]
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    class_wts = torch.tensor(weights, dtype=torch.float32).to(device)

    # Gamma=3.0 (vs 2.0 for base training) because fine-tuning datasets are
    # typically more imbalanced — ambient samples outnumber threat samples.
    criterion = FocalLoss(alpha=class_wts, gamma=3.0,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 2), T_mult=2)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(args.epochs):
        # Warmup phase transition: unfreeze CNN after warmup_epochs.
        # The optimizer is RESET with 10x lower LR to prevent large gradient
        # updates from destroying pretrained CNN filters. The scheduler is
        # also reset to give the newly unfrozen layers a fresh cosine cycle.
        if use_warmup and epoch == args.warmup_epochs:
            unfreeze_all()
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr / 10, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, (args.epochs - args.warmup_epochs) // 2), T_mult=2)
            print(f"  [Epoch {epoch+1}] CNN unfrozen, lr={args.lr/10:.1e}")

        model.train()
        run_loss = correct = total_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if np.random.random() < MIXUP_PROB:
                inputs, la, lb, lam = mixup_batch(inputs, labels, 0.3)
                outputs = model(inputs)
                loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
                c_lab = la if lam >= 0.5 else lb
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                c_lab = labels

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            correct += (outputs.detach().argmax(1) == c_lab).sum().item()
            total_samples += labels.size(0)

        scheduler.step(epoch)
        train_loss = run_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / total_samples

        # Quick val on test set for early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_inp, v_lab in test_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                val_loss += criterion(model(v_inp), v_lab).item()
        val_loss /= max(len(test_loader), 1)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            improved = " *"

        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% "
              f"VL={val_loss:.4f}{improved}")

    # ── Step 6: Evaluate fine-tuned model ────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    ft_auc, ft_acc, ft_f1 = evaluate_model(model, test_loader, device)

    # ── Step 7: Threshold optimization ───────────────────────────────
    print("\n  Optimizing detection threshold...")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import roc_curve, precision_recall_fscore_support
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    fpr, tpr, roc_thresholds = roc_curve(all_labels_np, all_probs_np)

    # Youden's J statistic: threshold that maximizes (TPR - FPR).
    # Gives a balanced operating point between sensitivity and specificity.
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    youden_thresh = float(roc_thresholds[j_idx])

    # F1-optimal: brute-force search over threshold grid.
    # Weighted F1 accounts for class imbalance in the test set.
    best_f1_thresh, best_f1_val = 0.5, 0.0
    for t in np.arange(0.3, 0.9, 0.01):
        preds = (all_probs_np >= t).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(
            all_labels_np, preds, average='weighted', zero_division=0)
        if f1_t > best_f1_val:
            best_f1_val = f1_t
            best_f1_thresh = float(t)

    # Low-FPR threshold: the highest threshold where FPR stays <= 5%.
    # This is the RECOMMENDED threshold for production deployment because
    # false alarms (wrongly triggering an alert) erode user trust far more
    # than occasional missed detections. Clamped to [0.3, 0.95] as a
    # safety net against degenerate ROC curves.
    low_fpr_thresh = 0.5
    for i in range(len(fpr)):
        if fpr[i] <= 0.05:
            low_fpr_thresh = float(roc_thresholds[i])
    low_fpr_thresh = max(0.3, min(0.95, low_fpr_thresh))

    thresholds = {
        'youden': youden_thresh,
        'f1_optimal': best_f1_thresh,
        'low_fpr': low_fpr_thresh,
    }

    print(f"  Youden's J:   {youden_thresh:.3f}")
    print(f"  F1-optimal:   {best_f1_thresh:.3f} (F1={best_f1_val:.3f})")
    print(f"  Low-FPR (5%): {low_fpr_thresh:.3f}")

    # ── Step 7b: Site-ambient threshold recalibration ─────────────────
    # The low_fpr above is calibrated on the combined test distribution
    # (universal test + ambient chunks). That distribution is not the
    # distribution the deployed model actually faces — a real metro
    # station has different speech/platform/announcement mix than the
    # universal test pool. Recalibrate on a held-out ambient directory
    # from the target site, using sliding-window inference (matches the
    # production inference.py semantics) and temporal-majority
    # aggregation. Emit the result as `low_fpr_site`, a separate key so
    # downstream code can choose between the two without losing either.
    if args.calibration_ambient_dir and os.path.isdir(args.calibration_ambient_dir):
        from safecommute.pipeline.test_deployment import (
            load_model_and_stats, sliding_window_inference, fires,
        )
        print(f"\n  Calibrating site threshold on: "
              f"{args.calibration_ambient_dir} "
              f"(majority_k={args.calibration_majority_k})")

        # Save a temporary checkpoint to disk so load_model_and_stats reuses
        # the exact same loader the deployment runtime uses.
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        tmp_ckpt = os.path.join(save_dir, f"{args.environment}_tmp_for_calib.pth")
        torch.save(model.state_dict(), tmp_ckpt)
        try:
            cal_model, cal_mean, cal_std = load_model_and_stats(tmp_ckpt)
            wavs = sorted(f for f in os.listdir(args.calibration_ambient_dir)
                          if f.endswith('.wav'))
            if not wavs:
                print("  WARNING: calibration dir contains no .wav files — "
                      "skipping site recalibration.")
                site_thresh = None
                sweep = []
            else:
                per_wav_probs = []
                for name in wavs:
                    path = os.path.join(args.calibration_ambient_dir, name)
                    import librosa
                    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                    per_wav_probs.append(
                        sliding_window_inference(cal_model, y, cal_mean,
                                                 cal_std, 0.5))

                candidate_thresholds = np.round(
                    np.arange(0.30, 0.951, 0.01), 3)
                sweep = []
                for t in candidate_thresholds:
                    fires_count = sum(
                        1 for probs in per_wav_probs
                        if fires(probs, float(t), args.calibration_majority_k))
                    fp = fires_count / len(per_wav_probs)
                    sweep.append({'threshold': float(t), 'fp_rate': fp})

                # Highest threshold where site-FP ≤ 5 % (keeps recall as
                # high as possible subject to the FP budget).
                site_thresh = None
                for row in sweep:
                    if row['fp_rate'] <= 0.05:
                        site_thresh = row['threshold']
                        break
                # Clamp to the same sanity band used for low_fpr.
                if site_thresh is not None:
                    site_thresh = float(max(0.30, min(0.95, site_thresh)))

            if site_thresh is not None:
                thresholds['low_fpr_site'] = site_thresh
                thresholds['low_fpr_site_majority_k'] = args.calibration_majority_k
                thresholds['low_fpr_site_calibration_dir'] = args.calibration_ambient_dir
                thresholds['low_fpr_site_sweep'] = sweep
                print(f"  Low-FPR site: {site_thresh:.3f} "
                      f"(k={args.calibration_majority_k}, "
                      f"n_wavs={len(wavs) if wavs else 0})")
            else:
                thresholds['low_fpr_site_sweep'] = sweep
                print(f"  Low-FPR site: NO threshold achieves site-FP ≤ 5% "
                      f"(k={args.calibration_majority_k}) — sweep saved anyway")
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)
    elif args.calibration_ambient_dir:
        print(f"\n  WARNING: --calibration-ambient-dir "
              f"'{args.calibration_ambient_dir}' not a directory; skipping "
              f"site recalibration.")

    # ── Step 8: Save ─────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    save_path = f"models/{args.environment}_model.pth"
    torch.save(model.state_dict(), save_path)

    thresh_path = f"models/{args.environment}_thresholds.json"
    with open(thresh_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    # ── Step 9: Comparison ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fine-tuning Results: {args.environment}")
    print(f"{'='*60}")
    print(f"  {'':20} {'AUC':>8} {'Acc':>8} {'F1':>8}")
    print(f"  {'Base model':<20} {base_auc:>8.4f} {base_acc:>8.4f} {base_f1:>8.4f}")
    print(f"  {'Fine-tuned':<20} {ft_auc:>8.4f} {ft_acc:>8.4f} {ft_f1:>8.4f}")
    print(f"  {'Delta':<20} {ft_auc - base_auc:>+8.4f} {ft_acc - base_acc:>+8.4f} "
          f"{ft_f1 - base_f1:>+8.4f}")
    print(f"\n  Saved model: {save_path}")
    print(f"  Saved thresholds: {thresh_path}")

    # Catastrophic forgetting guard: warn if threat detection AUC regressed
    # significantly. A drop >0.05 suggests the model is forgetting what
    # threats sound like. Remedies: --freeze-cnn, fewer epochs, or more
    # base unsafe samples in the fine-tuning mix.
    if ft_auc < base_auc - 0.05:
        print(f"\n  WARNING: AUC dropped by {base_auc - ft_auc:.3f}. "
              f"Consider --freeze-cnn or fewer epochs.")

    # ── Step 9: Cleanup ──────────────────────────────────────────────
    shutil.rmtree(tmp_dir)
    print(f"  Cleaned up temp directory")
    print("\nDone.")


if __name__ == "__main__":
    main()
