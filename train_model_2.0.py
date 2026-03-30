import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "prepared_data"
STATS_PATH      = "feature_stats.json"
BATCH_SIZE      = 32
EPOCHS          = 20           # more epochs + early stopping replaces a fixed low cap
LEARNING_RATE   = 5e-4         # slightly lower; scheduler will anneal from here
PATIENCE        = 5            # early stopping patience (val-loss based)
MODEL_SAVE_PATH = "safecommute_edge_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Fixed time dimension: ceil(TARGET_LENGTH / hop_length) = ceil(48000 / 256) = 188
# Every tensor is cropped or zero-padded to this width so the DataLoader can
# stack them into a batch without shape mismatches.
TIME_FRAMES = 188


class TensorAudioDataset(Dataset):
    def __init__(self, split_dir, mean=0.0, std=1.0):
        self.filepaths = []
        self.labels    = []
        self.mean      = mean
        self.std       = std

        for label, class_name in enumerate(['0_safe', '1_unsafe']):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        features = torch.load(self.filepaths[idx], weights_only=True)

        # ── Enforce fixed time dimension ──────────────────────────────────
        # Shape is [1, n_mels, T]; T may vary slightly across clips.
        t = features.shape[-1]
        if t > TIME_FRAMES:
            features = features[:, :, :TIME_FRAMES]          # crop
        elif t < TIME_FRAMES:
            pad = torch.zeros(1, features.shape[1], TIME_FRAMES - t)
            features = torch.cat([features, pad], dim=-1)    # right-pad

        # ── z-score normalisation ─────────────────────────────────────────
        features = (features - self.mean) / (self.std + 1e-8)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SafeCommuteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v2(weights='DEFAULT')

        # ── Adapt first conv for 1-channel Mel input ──────────────────────
        # Instead of random re-init, average the pretrained 3-channel weights
        # across the channel dimension. This preserves learned edge/texture
        # detectors rather than discarding them entirely.
        original_conv    = self.backbone.features[0][0]
        averaged_weights = original_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        new_conv.weight.data = averaged_weights
        self.backbone.features[0][0] = new_conv

        # ── Binary output head ─────────────────────────────────────────────
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_stats():
    """Load pre-computed feature normalisation statistics."""
    if not os.path.exists(STATS_PATH):
        print(f"Warning: '{STATS_PATH}' not found. Running without normalisation.")
        print("  → Re-run data_pipeline.py to generate this file.")
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean, std = stats['mean'], stats['std']
    print(f"Loaded feature stats: mean={mean:.2f}, std={std:.2f}")
    return mean, std


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights so the loss treats
    an imbalanced dataset fairly (common in RAVDESS: 3 safe emotions, 2 unsafe).
    """
    counts = [0, 0]
    for label in dataset.labels:
        counts[label] += 1
    total = sum(counts)
    # weight = total / (n_classes * count_per_class)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class distribution — Safe: {counts[0]}, Unsafe: {counts[1]}")
    print(f"  Class weights      — Safe: {weights[0]:.3f}, Unsafe: {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    mean, std = load_stats()

    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset   = TensorAudioDataset(os.path.join(DATA_DIR, 'val'),   mean, std)

    if len(train_dataset) == 0:
        print("Error: No training data found. Run data_pipeline.py first.")
        return

    print(f"\nDataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Halve LR when val loss plateaus for 3 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss   = float('inf')
    epochs_no_impro = 0

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
    print("-" * 50)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        run_loss = correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            preds     = outputs.detach().argmax(1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
        train_acc  = 100 * correct / total
        train_loss = run_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = v_correct = v_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out  = model(v_inp)
                val_loss += criterion(v_out, v_lab).item()
                v_preds   = v_out.argmax(1)
                v_correct += (v_preds == v_lab).sum().item()
                v_total   += v_lab.size(0)
                all_preds.extend(v_preds.cpu().tolist())
                all_labels.extend(v_lab.cpu().tolist())

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc      = 100 * v_correct / v_total if v_total > 0 else 0

        print(f"{epoch+1:>6} {train_loss:>11.4f} {train_acc:>9.2f}% "
              f"{val_loss_avg:>9.4f} {val_acc:>7.2f}%")

        scheduler.step(val_loss_avg)

        # ── Save best checkpoint ──
        if val_loss_avg < best_val_loss:
            best_val_loss   = val_loss_avg
            epochs_no_impro = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"         ↑ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # ── Final evaluation on validation set ──
    print("\n" + "=" * 50)
    print(" FINAL VALIDATION METRICS (best checkpoint)")
    print("=" * 50)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for v_inp, v_lab in val_loader:
            v_inp = v_inp.to(device)
            preds = model(v_inp).argmax(1).cpu().tolist()
            final_preds.extend(preds)
            final_labels.extend(v_lab.tolist())

    print(classification_report(final_labels, final_preds, target_names=['Safe', 'Unsafe']))
    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':10} {'Pred Safe':>10} {'Pred Unsafe':>12}")
    print(f"  {'Actual Safe':10} {cm[0][0]:>10} {cm[0][1]:>12}")
    print(f"  {'Actual Unsafe':10} {cm[1][0]:>10} {cm[1][1]:>12}")

    print(f"\nModel saved to '{MODEL_SAVE_PATH}'.")
    print("Ready for edge inference.")


if __name__ == "__main__":
    train()