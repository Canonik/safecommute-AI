"""
Experiment 0: Baseline evaluation.
Re-evaluate the current best model to establish ground truth for comparison.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH
from research.experiments.eval_utils import full_evaluation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    print("=== Baseline Evaluation ===")
    results, breakdown = full_evaluation(model, device, "Baseline (production model)", "AUC=0.950 target")
    return results


if __name__ == "__main__":
    main()
