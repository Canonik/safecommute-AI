"""
Experiment 4: Temperature Scaling.
Learn a single temperature parameter T on validation set.
Improves confidence calibration: softmax(logits / T).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, DATA_DIR
from research.experiments.eval_utils import (
    load_stats, get_test_loader, get_train_val_loaders,
    per_source_breakdown, measure_latency,
    count_parameters, model_size_mb, log_experiment
)


class TemperatureScaler(nn.Module):
    """Wraps a model with a learned temperature parameter."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature


def learn_temperature(model, val_loader, device, max_iter=50, lr=0.01):
    """Optimize temperature on validation set using NLL loss."""
    model.eval()

    # Collect all logits and labels from validation set
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs.to(device))
            all_logits.append(logits)
            all_labels.append(labels.to(device))

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    print("=== Temperature Scaling ===")

    # Learn temperature on validation set
    _, _, _, val_loader = get_train_val_loaders()
    optimal_T = learn_temperature(model, val_loader, device)
    print(f"  Optimal temperature: T={optimal_T:.4f}")

    # Evaluate on test set with temperature scaling
    test_dataset, test_loader = get_test_loader()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            scaled_logits = logits / optimal_T
            probs = torch.softmax(scaled_logits, dim=1)[:, 1]
            preds = scaled_logits.argmax(1)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    results = {
        'auc': roc_auc_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
    }
    breakdown = per_source_breakdown(test_dataset, all_preds, all_labels)
    lat_mean, lat_std = measure_latency(model, device)
    params = count_parameters(model)
    size = model_size_mb(model)

    log_experiment(
        "Temperature Scaling", results, breakdown,
        lat_mean, lat_std, params, size,
        f"T={optimal_T:.3f}, learned on val set"
    )

    print(f"  AUC={results['auc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")
    for src, data in sorted(breakdown.items()):
        print(f"    {src}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    main()
