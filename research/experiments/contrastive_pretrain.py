"""
Experiment 10: Contrastive Pre-training (SimCLR-style).
Create augmented pairs of spectrograms, train encoder to pull same-audio
augmentations together and push different ones apart. Then fine-tune with labels.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

from safecommute.model import SafeCommuteCNN, ConvBlock, SEBlock
from safecommute.constants import DATA_DIR, N_MELS
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, full_evaluation
)
from v_3.train_experimental import FocalLoss, spec_augment_strong, mixup_batch, compute_class_weights

BATCH_SIZE = 64
PRETRAIN_EPOCHS = 15
FINETUNE_EPOCHS = 20
LEARNING_RATE_PRETRAIN = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
PATIENCE = 6
TEMPERATURE = 0.07


def augment_spectrogram(tensor):
    """Create a random augmented view of a spectrogram."""
    import torchaudio.transforms as T

    aug = tensor.clone()

    # Random frequency masking
    if random.random() < 0.8:
        mask_param = random.randint(5, 15)
        aug = T.FrequencyMasking(freq_mask_param=mask_param)(aug)

    # Random time masking
    if random.random() < 0.8:
        mask_param = random.randint(10, 30)
        aug = T.TimeMasking(time_mask_param=mask_param)(aug)

    # Additive noise
    if random.random() < 0.5:
        noise = torch.randn_like(aug) * random.uniform(0.05, 0.2)
        aug = aug + noise

    # Random time shift
    if random.random() < 0.5:
        shift = random.randint(-15, 15)
        aug = torch.roll(aug, shifts=shift, dims=-1)

    # Random frequency shift
    if random.random() < 0.3:
        shift = random.randint(-3, 3)
        aug = torch.roll(aug, shifts=shift, dims=-2)

    return aug


class ContrastiveDataset(Dataset):
    """Dataset that returns two augmented views of each sample."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        features, label = self.base[idx]
        view1 = augment_spectrogram(features)
        view2 = augment_spectrogram(features)
        return view1, view2, label


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ContrastiveEncoder(nn.Module):
    """SafeCommute encoder + projection head for contrastive pre-training."""
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)

        freq_after_blocks = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        self.projection = ProjectionHead(384, 256, 128)

    def encode(self, x):
        """Get representation before projection."""
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))

        out, h = self.gru(x)
        h_last = h.squeeze(0)
        h_mean = out.mean(dim=1)
        h_max = out.max(dim=1).values

        return torch.cat([h_last, h_mean, h_max], dim=1)  # (B, 384)

    def forward(self, x):
        """Get projected representation for contrastive loss."""
        rep = self.encode(x)
        return self.projection(rep)


def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def pretrain_contrastive(save_path):
    """Phase 1: Contrastive pre-training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_stats()
    base_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    contrastive_dataset = ContrastiveDataset(base_dataset)
    loader = DataLoader(contrastive_dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)

    encoder = ContrastiveEncoder().to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE_PRETRAIN, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        total_loss = 0
        n_batches = 0

        for view1, view2, _ in loader:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = encoder(view1)
            z2 = encoder(view2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        print(f"  Pretrain E{epoch+1:>2} Loss={avg_loss:.4f} LR={scheduler.get_last_lr()[0]:.6f}")

    torch.save(encoder.state_dict(), save_path)
    return encoder


def finetune_from_pretrained(encoder_path, save_path):
    """Phase 2: Fine-tune with labels using pretrained encoder weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create full model and load pretrained encoder weights
    model = SafeCommuteCNN().to(device)

    # Load pretrained encoder
    encoder = ContrastiveEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))

    # Transfer encoder weights to SafeCommuteCNN
    model.bn_input.load_state_dict(encoder.bn_input.state_dict())
    model.block1.load_state_dict(encoder.block1.state_dict())
    model.block2.load_state_dict(encoder.block2.state_dict())
    model.block3.load_state_dict(encoder.block3.state_dict())
    model.freq_reduce.load_state_dict(encoder.freq_reduce.state_dict())
    model.gru.load_state_dict(encoder.gru.state_dict())
    print("  Transferred pretrained encoder weights")

    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        run_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            aug_inputs = []
            for i in range(inputs.size(0)):
                aug_inputs.append(spec_augment_strong(inputs[i].cpu()).to(device))
            inputs = torch.stack(aug_inputs)

            if np.random.random() < 0.5:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, alpha=0.3)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                c_lab = labels_a if lam >= 0.5 else labels_b
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                c_lab = labels

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = outputs.detach().argmax(1)
            run_loss += loss.item()
            correct += (preds == c_lab).sum().item()
            total += labels.size(0)

        train_loss = run_loss / len(train_loader)
        train_acc = 100 * correct / total

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
        val_acc = 100 * v_correct / v_total

        scheduler.step(epoch)
        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

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

    return model


def main():
    seed_everything()
    os.makedirs("research/results", exist_ok=True)

    encoder_path = "research/results/contrastive_encoder.pth"
    model_path = "research/results/contrastive_finetuned.pth"

    print("=== Contrastive Pre-training (SimCLR-style) ===")

    # Phase 1: Contrastive pre-training
    print("\n--- Phase 1: Contrastive Pre-training ---")
    pretrain_contrastive(encoder_path)

    # Phase 2: Fine-tune with labels
    print("\n--- Phase 2: Fine-tuning ---")
    finetune_from_pretrained(encoder_path, model_path)

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    full_evaluation(model, device, "Contrastive Pretrain + Finetune",
                   "SimCLR 15ep pretrain + 20ep finetune")


if __name__ == "__main__":
    main()
