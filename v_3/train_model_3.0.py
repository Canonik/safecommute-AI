import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import (
    DATA_DIR, STATS_PATH, THRESHOLDS_PATH, MODEL_SAVE_PATH,
    N_MELS, TIME_FRAMES,
)
from safecommute.utils import seed_everything

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 25
LEARNING_RATE   = 3e-4
PATIENCE        = 6
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.3
MIXUP_PROB      = 0.5

# Distillation hyperparameters
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA       = 0.7   # weight for soft (teacher) loss


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    return x_m, y, y[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_stats():
    if not os.path.exists(STATS_PATH):
        print(f"Warning: '{STATS_PATH}' not found. Running without normalisation.")
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    print(f"Feature stats: mean={s['mean']:.2f}, std={s['std']:.2f}")
    return s['mean'], s['std']


def compute_class_weights(dataset):
    counts = [0, 0]
    for lbl in dataset.labels:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class distribution — Safe: {counts[0]}, Unsafe: {counts[1]}")
    print(f"  Class weights      — Safe: {weights[0]:.3f}, Unsafe: {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def calibrate_and_save_thresholds(model, val_loader, device, target_fpr=0.05):
    print("\nCalibrating decision thresholds on validation set…")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for v_inp, v_lab in val_loader:
            probs = torch.softmax(model(v_inp.to(device)), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(v_lab.tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    fpr, tpr, thresholds = roc_curve(labels_arr, probs_arr)

    valid = fpr <= target_fpr
    if valid.any():
        red_thresh = float(thresholds[valid][-1])
    else:
        red_thresh = 0.70

    amber_thresh = round(0.5 + (red_thresh - 0.5) / 2, 3)
    red_thresh = round(red_thresh, 3)

    f1_scores = []
    for t in thresholds:
        preds = (probs_arr >= t).astype(int)
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_scores.append(2 * prec * rec / (prec + rec + 1e-8))
    f1_thresh = round(float(thresholds[np.argmax(f1_scores)]), 3)

    result = {
        "amber": amber_thresh,
        "red": red_thresh,
        "f1_optimal": f1_thresh,
        "note": (f"Calibrated at target_fpr={target_fpr}. "
                 f"Red fires when FPR≤{target_fpr*100:.0f}% on val set.")
    }
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Amber threshold : {amber_thresh}")
    print(f"  Red threshold   : {red_thresh}  (FPR ≤ {target_fpr*100:.0f}% on val)")
    print(f"  F1-optimal      : {f1_thresh}")
    print(f"  Saved → {THRESHOLDS_PATH}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label='ROC curve')
        plt.axvline(x=target_fpr, color='r', linestyle='--',
                    label=f'Target FPR={target_fpr}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SafeCommute AI — Validation ROC')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=120)
        print("  ROC curve saved → roc_curve.png")
        plt.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(distill=False):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if distill:
        print("Knowledge distillation: ENABLED (T={}, α={})".format(
            DISTILL_TEMPERATURE, DISTILL_ALPHA))
    print()

    mean, std = load_stats()

    train_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'train'), mean, std, load_teacher=distill)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)

    if len(train_dataset) == 0:
        print("Error: No training data found. Run data_pipeline_3.0.py first.")
        return

    print(f"Dataset — Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_wts, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    print(f"\nTraining up to {EPOCHS} epochs  |  early-stop patience={PATIENCE}\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
    print("-" * 52)

    for epoch in range(EPOCHS):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        run_loss = correct = total = 0

        for batch in train_loader:
            if distill:
                inputs, labels, teacher_soft = batch
                teacher_soft = teacher_soft.to(device)
            else:
                inputs, labels = batch

            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup augmentation (skip during distillation for simplicity)
            use_mixup = not distill and np.random.random() < MIXUP_PROB
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                preds = outputs.detach().argmax(1)
                c_lab = labels_a if lam >= 0.5 else labels_b
            else:
                outputs = model(inputs)
                loss_hard = criterion(outputs, labels)

                if distill:
                    # Knowledge distillation loss
                    T = DISTILL_TEMPERATURE
                    valid_teacher = teacher_soft[:, 0] >= 0  # filter sentinel values
                    if valid_teacher.any():
                        s_logits = outputs[valid_teacher]
                        t_soft = teacher_soft[valid_teacher]
                        student_log = F.log_softmax(s_logits / T, dim=1)
                        teacher_prob = F.softmax(t_soft / T, dim=1)
                        loss_soft = F.kl_div(
                            student_log, teacher_prob,
                            reduction='batchmean') * (T ** 2)
                        loss = DISTILL_ALPHA * loss_soft + (1 - DISTILL_ALPHA) * loss_hard
                    else:
                        loss = loss_hard
                else:
                    loss = loss_hard

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

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out = model(v_inp)
                val_loss += criterion(v_out, v_lab).item()
                v_preds = v_out.argmax(1)
                v_correct += (v_preds == v_lab).sum().item()
                v_total += v_lab.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        print(f"{epoch+1:>6} {train_loss:>11.4f} {train_acc:>9.2f}% "
              f"{val_loss_avg:>9.4f} {val_acc:>7.2f}%")

        scheduler.step(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"         ↑ Best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= PATIENCE:
                print(f"\nEarly stopping after epoch {epoch + 1}.")
                break

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(" FINAL VALIDATION METRICS (best checkpoint)")
    print("=" * 52)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for v_inp, v_lab in val_loader:
            preds = model(v_inp.to(device)).argmax(1).cpu().tolist()
            final_preds.extend(preds)
            final_labels.extend(v_lab.tolist())

    print(classification_report(final_labels, final_preds, target_names=['Safe', 'Unsafe']))
    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion matrix:")
    print(f"  {'':12} {'Pred Safe':>10} {'Pred Unsafe':>12}")
    print(f"  {'Actual Safe':12} {cm[0][0]:>10} {cm[0][1]:>12}")
    print(f"  {'Actual Unsafe':12} {cm[1][0]:>10} {cm[1][1]:>12}")

    # ── Test set evaluation (if available) ────────────────────────────────────
    test_dir = os.path.join(DATA_DIR, 'test')
    if os.path.exists(test_dir):
        test_dataset = TensorAudioDataset(test_dir, mean, std)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True)
            test_preds, test_labels = [], []
            with torch.no_grad():
                for t_inp, t_lab in test_loader:
                    preds = model(t_inp.to(device)).argmax(1).cpu().tolist()
                    test_preds.extend(preds)
                    test_labels.extend(t_lab.tolist())

            print("\n" + "=" * 52)
            print(" HELD-OUT TEST SET METRICS")
            print("=" * 52)
            print(classification_report(
                test_labels, test_preds, target_names=['Safe', 'Unsafe']))
            cm_test = confusion_matrix(test_labels, test_preds)
            print("Confusion matrix (test):")
            print(f"  {'':12} {'Pred Safe':>10} {'Pred Unsafe':>12}")
            print(f"  {'Actual Safe':12} {cm_test[0][0]:>10} {cm_test[0][1]:>12}")
            print(f"  {'Actual Unsafe':12} {cm_test[1][0]:>10} {cm_test[1][1]:>12}")

    # ── Threshold calibration ─────────────────────────────────────────────────
    calibrate_and_save_thresholds(model, val_loader, device, target_fpr=0.05)
    print(f"\nModel → '{MODEL_SAVE_PATH}'  |  Thresholds → '{THRESHOLDS_PATH}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SafeCommute AI model")
    parser.add_argument('--distill', action='store_true',
                        help='Enable knowledge distillation from pre-computed teacher labels')
    args = parser.parse_args()
    train(distill=args.distill)
