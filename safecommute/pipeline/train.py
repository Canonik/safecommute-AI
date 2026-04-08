"""
Training script for SafeCommute AI.

Trains the SafeCommuteCNN on Layer 1 (universal threats) + Layer 2 (hard
negatives) data prepared by data_pipeline.py. The model learns a universal
safe/unsafe boundary that can later be fine-tuned per deployment (finetune.py).

Two-layer augmentation strategy (both operate on clean .pt spectrograms):
  Layer 1 — Base augmentation (dataset.py __getitem__, always on):
    SpecAugment frequency and time masking, 50% probability each.
    Simulates partial occlusion of spectral features (e.g., a loud background
    sound masking part of a scream). Lightweight, per-sample, CPU-side.

  Layer 2 — Strong augmentation (--strong-aug flag, this file):
    Batch-level GPU ops applied on top of base augmentation:
      - Gaussian noise (30% of samples, sigma=0.1): simulates microphone noise
      - Circular time shift (30%, +/-20 frames): simulates event timing jitter
      - Frequency band dropout (20%, 3-10 bins): simulates missing freq bands
    These are heavier transforms that benefit from GPU parallelism.

Loss function choices:
  - Focal loss (--focal): down-weights easy examples, focuses on hard ones.
    Critical because most safe samples are trivially easy (silence, music),
    while the boundary cases (laughter vs. screaming) are rare but important.
  - Gamma parameter (--gamma): controls how aggressively easy examples are
    down-weighted. Default 2.0; experiments showed gamma=3.0 helps when the
    dataset is highly imbalanced (see experiment_log.md).
  - Label smoothing (0.1): prevents overconfident predictions, improves
    calibration of probability outputs used for threshold-based deployment.
  - Mixup (alpha=0.3, 50% probability): interpolates pairs of training samples
    and labels, acting as a regularizer that smooths the decision boundary.
"""
import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH, THRESHOLDS_PATH, TIME_FRAMES
from safecommute.utils import seed_everything, worker_init_fn

# Training hyperparameters — tuned via grid search (see experiment_log.md)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4       # AdamW default works well; higher LRs diverge with focal loss
PATIENCE = 6               # Early stopping patience (epochs without val loss improvement)
LABEL_SMOOTHING = 0.1      # Prevents overconfident logits, improves threshold calibration
MIXUP_ALPHA = 0.3          # Beta distribution parameter for mixup interpolation weight
MIXUP_PROB = 0.5           # Apply mixup to 50% of batches

# Per-source prefix mappings for detailed evaluation
SAFE_SOURCES = {
    'as_laughter': 'as_laughter',
    'as_crowd': 'as_crowd',
    'as_speech': 'as_speech',
    'yt_metro': 'yt_metro',
}
UNSAFE_SOURCES = {
    'as_screaming': 'as_screaming',
    'as_shout': 'as_shout',
    'as_yell': 'as_yell',
    'yt_scream': 'yt_scream',
}

# Safe hard negatives that frequently trigger false alarms in deployment-like audio
HARD_NEGATIVE_SOURCES = {
    'as_laughter',
    'as_crowd',
    'as_speech',
    'yt_metro',
}

# Environmental noise injection parameters
NOISE_INJECT_PROB = 0.3    # Probability of injecting noise into a sample
NOISE_SNR_MIN = 0.0        # Minimum SNR in dB (loudest noise)
NOISE_SNR_MAX = 20.0       # Maximum SNR in dB (quietest noise)


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for handling class imbalance and hard examples.

    Standard cross-entropy treats all samples equally, but our dataset has many
    trivially easy safe samples (silence, music) and fewer hard boundary cases
    (laughter that sounds like screaming). Focal loss multiplies the CE loss by
    (1 - p_t)^gamma, where p_t is the model's predicted probability for the
    correct class. This down-weights easy examples (high p_t) and focuses
    training on hard examples (low p_t).

    Args:
        alpha: Per-class weight tensor (inverse frequency weighting).
        gamma: Focusing parameter. gamma=0 recovers standard CE. Higher gamma
               focuses more aggressively on hard examples. We use 2.0 for base
               training and 3.0 for fine-tuning (where class imbalance is worse).
        label_smoothing: Soft target smoothing factor.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                  reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        # (1 - pt)^gamma: easy examples (pt close to 1) get near-zero weight
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    """
    Mixup augmentation (Zhang et al., 2018): linearly interpolate pairs of
    training samples and their labels. The loss is computed as a weighted
    combination of losses for both original labels.

    Returns both label sets and the interpolation weight so the caller
    can compute: loss = lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b).
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    return x_m, y, y[idx], lam


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights for the loss function.

    With typical data ratios of ~3:1 safe:unsafe, the unsafe class gets a
    higher weight to prevent the model from achieving low loss by simply
    predicting everything as safe.
    """
    counts = [0, 0]
    for lbl in dataset.labels:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def classify_source(filename):
    """Extract source prefix from a .pt filename for per-source evaluation."""
    base = os.path.basename(filename)
    for prefix in sorted(list(SAFE_SOURCES.keys()) + list(UNSAFE_SOURCES.keys()), key=len, reverse=True):
        if base.startswith(prefix):
            return prefix
    return 'other'


def build_targeted_sampler(dataset, hard_neg_quota):
    """
    Build a weighted sampler that increases the sampling frequency of hard
    negative safe examples to approximately `hard_neg_quota` of drawn samples.
    """
    if hard_neg_quota <= 0.0 or hard_neg_quota >= 1.0 or len(dataset) == 0:
        return None, 0, 0, 1.0

    weights = np.ones(len(dataset), dtype=np.float64)
    hard_indices = []
    for i, path in enumerate(dataset.filepaths):
        if dataset.labels[i] != 0:
            continue
        source = classify_source(os.path.basename(path))
        if source in HARD_NEGATIVE_SOURCES:
            hard_indices.append(i)

    n_hard = len(hard_indices)
    n_total = len(dataset)
    if n_hard == 0 or n_hard == n_total:
        return None, n_hard, n_total, 1.0

    n_other = n_total - n_hard
    target_ratio = hard_neg_quota / max(1e-8, (1.0 - hard_neg_quota))
    multiplier = float(max(1.0, (target_ratio * n_other) / n_hard))
    for i in hard_indices:
        weights[i] = multiplier

    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_total,
        replacement=True,
    )
    return sampler, n_hard, n_total, multiplier


def per_source_accuracy(test_dir, model, device, mean, std):
    """Compute per-source accuracy breakdown on test set."""
    source_correct = defaultdict(int)
    source_total = defaultdict(int)

    for split_dir in ['0_safe', '1_unsafe']:
        label = int(split_dir[0])
        split_path = os.path.join(test_dir, split_dir)
        if not os.path.isdir(split_path):
            continue
        for fname in os.listdir(split_path):
            if not fname.endswith('.pt') or '_teacher.pt' in fname:
                continue
            source = classify_source(fname)
            fpath = os.path.join(split_path, fname)
            spec = torch.load(fpath, map_location='cpu', weights_only=True)
            if isinstance(spec, dict):
                spec = spec['spectrogram'] if 'spectrogram' in spec else list(spec.values())[0]
            # Expected shape: [1, 64, 188] -> need [1, 1, 64, 188]
            if spec.dim() == 2:  # [64, 188]
                spec = spec.unsqueeze(0).unsqueeze(0)
            elif spec.dim() == 3:  # [1, 64, 188]
                spec = spec.unsqueeze(0)
            elif spec.dim() != 4:
                continue  # skip unexpected shapes
            spec = (spec - mean) / (std + 1e-7)
            with torch.no_grad():
                logits = model(spec.to(device))
                pred = logits.argmax(1).item()
            source_correct[source] += int(pred == label)
            source_total[source] += 1

    results = {}
    for source in sorted(source_total.keys()):
        acc = source_correct[source] / source_total[source] if source_total[source] > 0 else 0
        results[source] = (acc, source_correct[source], source_total[source])
    return results


def load_noise_bank(data_dir, mean, std, noise_sources=None):
    """
    Pre-load ambient safe spectrograms from training set as a noise bank.
    These are real ambient recordings used to inject environmental
    noise during training, teaching the model to detect threats in noisy
    conditions.

    Returns a list of normalized spectrogram tensors on CPU.
    """
    if noise_sources is None:
        noise_sources = ['yt_metro']

    noise_dir = os.path.join(data_dir, 'train', '0_safe')
    noise_paths = []
    source_counts = {}
    for source in noise_sources:
        source_paths = sorted(glob.glob(os.path.join(noise_dir, f'{source}*.pt')))
        noise_paths.extend(source_paths)
        source_counts[source] = len(source_paths)
    noise_bank = []
    for p in noise_paths:
        spec = torch.load(p, map_location='cpu', weights_only=True)
        if isinstance(spec, dict):
            spec = spec.get('spectrogram', list(spec.values())[0])
        # Enforce shape [1, 64, TIME_FRAMES]
        while spec.dim() < 3:
            spec = spec.unsqueeze(0)
        if spec.dim() > 3:
            spec = spec.squeeze(0)
        t = spec.shape[-1]
        if t > TIME_FRAMES:
            spec = spec[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            pad = torch.zeros(spec.shape[0], spec.shape[1], TIME_FRAMES - t)
            spec = torch.cat([spec, pad], dim=-1)
        # Normalize same as training data
        spec = (spec - mean) / (std + 1e-8)
        noise_bank.append(spec)
    source_msg = ", ".join(f"{k}={v}" for k, v in source_counts.items())
    print(f"  Noise bank: {len(noise_bank)} samples loaded ({source_msg})")
    return noise_bank


def inject_noise(inputs, noise_bank, device, prob=NOISE_INJECT_PROB,
                 snr_min=NOISE_SNR_MIN, snr_max=NOISE_SNR_MAX):
    """
    Environmental noise injection: with probability `prob`, add a random
    yt_metro spectrogram to each sample at a random SNR level.

    Formula: noisy = clean + noise * 10^(-snr_db/20)
    where snr_db ~ Uniform(snr_min, snr_max)

    Higher SNR = quieter noise. At SNR=0dB the noise is as loud as the signal.
    At SNR=20dB the noise is 10x quieter than the signal.
    """
    if len(noise_bank) == 0:
        return inputs
    B = inputs.size(0)
    for i in range(B):
        if random.random() < prob:
            # Pick a random noise sample
            noise_spec = noise_bank[random.randint(0, len(noise_bank) - 1)]
            noise_spec = noise_spec.to(device)
            # Random SNR
            snr_db = random.uniform(snr_min, snr_max)
            scale = 10.0 ** (-snr_db / 20.0)
            # Add noise (noise_spec is [1, 64, 188], inputs[i] is [1, 64, 188])
            inputs[i] = inputs[i] + noise_spec * scale
    return inputs


def train(use_focal=False, use_cosine=False, use_strong_aug=False, gamma=2.0,
          label_smoothing=0.0, save_path=None, seed=42, epochs=None,
          mixup_alpha=None, mixup_prob=None, noise_inject=False,
          noise_sources=None, noise_snr_min=None, noise_snr_max=None,
          hard_neg_quota=0.0):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = save_path or MODEL_SAVE_PATH
    max_epochs = epochs or EPOCHS
    mx_alpha = mixup_alpha if mixup_alpha is not None else MIXUP_ALPHA
    mx_prob = mixup_prob if mixup_prob is not None else MIXUP_PROB
    snr_min = noise_snr_min if noise_snr_min is not None else NOISE_SNR_MIN
    snr_max = noise_snr_max if noise_snr_max is not None else NOISE_SNR_MAX

    mean, std = load_stats()

    # Load noise bank if noise injection is enabled
    noise_bank = []
    if noise_inject:
        noise_bank = load_noise_bank(DATA_DIR, mean, std, noise_sources=noise_sources)

    # Training dataset gets base augmentation (SpecAugment in __getitem__)
    train_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'train'), mean, std, augment=True)
    val_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'val'), mean, std, augment=False)

    if len(train_dataset) == 0:
        print("Error: No training data.")
        return None

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Focal={use_focal}(gamma={gamma}), Cosine={use_cosine}, "
          f"StrongAug={use_strong_aug}, Seed={seed}, Epochs={max_epochs}")
    print(f"Mixup: alpha={mx_alpha}, prob={mx_prob}")
    if noise_inject:
        print(f"Noise injection: prob={NOISE_INJECT_PROB}, SNR=[{snr_min},{snr_max}]dB")
        if noise_sources:
            print(f"Noise sources: {', '.join(noise_sources)}")

    sampler, n_hard, n_total, hard_mult = build_targeted_sampler(train_dataset, hard_neg_quota)
    if sampler is not None:
        print(f"Hard-negative quota: target={hard_neg_quota:.2f}, "
              f"hard_neg={n_hard}/{n_total}, weight_boost={hard_mult:.2f}x")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)

    if use_focal:
        criterion = FocalLoss(alpha=class_wts, gamma=gamma,
                              label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_wts,
                                        label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Cosine annealing with warm restarts (T_0=5, T_mult=2) gives restart
    # cycles at epochs 5, 15, 35... This periodically "reheats" the LR,
    # helping escape local minima. Alternative: ReduceLROnPlateau for
    # more conservative, loss-driven LR decay.
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(max_epochs):
        model.train()
        run_loss = correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Strong augmentation: per-sample GPU ops (no CPU transfer)
            if use_strong_aug:
                B = inputs.size(0)
                # Gaussian noise: per-sample mask (30% of samples)
                noise_mask = torch.rand(B, 1, 1, 1, device=inputs.device) < 0.3
                inputs = inputs + noise_mask * torch.randn_like(inputs) * 0.1
                # Circular time shift: per-sample random shift (30% of samples)
                for i in range(B):
                    if random.random() < 0.3:
                        shift = random.randint(-20, 20)
                        inputs[i] = torch.roll(inputs[i], shifts=shift, dims=-1)
                # Frequency band dropout: per-sample (20% of samples)
                for i in range(B):
                    if random.random() < 0.2:
                        f_start = random.randint(0, 50)
                        f_width = random.randint(3, 10)
                        inputs[i, :, f_start:f_start + f_width, :] = 0

            # Environmental noise injection: add ambient noise at random SNR
            if noise_inject and len(noise_bank) > 0:
                inputs = inject_noise(inputs, noise_bank, device,
                                      snr_min=snr_min, snr_max=snr_max)

            if np.random.random() < mx_prob:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, alpha=mx_alpha)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                preds = outputs.detach().argmax(1)
                c_lab = labels_a if lam >= 0.5 else labels_b
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.detach().argmax(1)
                c_lab = labels

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            correct += (preds == c_lab).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = run_loss / len(train_loader)

        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out = model(v_inp)
                val_loss += criterion(v_out, v_lab).item()
                v_correct += (v_out.argmax(1) == v_lab).sum().item()
                v_total += v_lab.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        if use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss_avg)

        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% "
              f"VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            torch.save(model.state_dict(), save_path)
            print(" *")
        else:
            epochs_no_impro += 1
            print()
            if epochs_no_impro >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Evaluate AUC on test set
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'test'), mean, std, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for t_inp, t_lab in test_loader:
            logits = model(t_inp.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(t_lab.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n  TEST: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # Per-source accuracy breakdown
    test_dir = os.path.join(DATA_DIR, 'test')
    source_results = per_source_accuracy(test_dir, model, device, mean, std)
    print("\n  Per-source accuracy:")
    for source, (s_acc, s_corr, s_tot) in source_results.items():
        print(f"    {source:20s}: {s_acc*100:5.1f}% ({s_corr}/{s_tot})")

    return {'auc': auc, 'acc': acc, 'f1': f1, 'per_source': source_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--focal', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--strong-aug', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0, was 0.1 in v1)')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Max training epochs (default: 25)')
    parser.add_argument('--mixup-alpha', type=float, default=None,
                        help='Mixup alpha (default: 0.3)')
    parser.add_argument('--mixup-prob', type=float, default=None,
                        help='Mixup probability (default: 0.5)')
    parser.add_argument('--noise-inject', action='store_true',
                        help='Enable environmental noise injection using safe ambient sources')
    parser.add_argument('--noise-sources', type=str, default='yt_metro',
                        help='Comma-separated safe source prefixes for noise bank '
                             '(e.g. yt_metro,yt_bar,yt_bus)')
    parser.add_argument('--noise-snr-min', type=float, default=NOISE_SNR_MIN,
                        help='Minimum SNR for injected noise (dB)')
    parser.add_argument('--noise-snr-max', type=float, default=NOISE_SNR_MAX,
                        help='Maximum SNR for injected noise (dB)')
    parser.add_argument('--hard-neg-quota', type=float, default=0.0,
                        help='Target minibatch fraction for safe hard negatives '
                             '(as_laughter/as_crowd/as_speech/yt_metro). 0 disables.')
    args = parser.parse_args()
    noise_sources = [s.strip() for s in args.noise_sources.split(',') if s.strip()]
    train(use_focal=args.focal, use_cosine=args.cosine, use_strong_aug=args.strong_aug,
          gamma=args.gamma, label_smoothing=args.label_smoothing,
          save_path=args.save, seed=args.seed, epochs=args.epochs,
          mixup_alpha=args.mixup_alpha, mixup_prob=args.mixup_prob,
          noise_inject=args.noise_inject, noise_sources=noise_sources,
          noise_snr_min=args.noise_snr_min, noise_snr_max=args.noise_snr_max,
          hard_neg_quota=args.hard_neg_quota)
