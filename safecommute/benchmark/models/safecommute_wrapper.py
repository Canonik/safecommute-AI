"""Wrapper for SafeCommute CNN model (original + quantized variants)."""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.features import preprocess
from safecommute.constants import (
    MODEL_SAVE_PATH, STATS_PATH, N_MELS, TIME_FRAMES,
)


class SafeCommuteWrapper:
    """Wraps SafeCommuteCNN for benchmarking."""

    def __init__(self, quantized=False):
        self.quantized = quantized
        self.model = None
        self.feat_mean = 0.0
        self.feat_std = 1.0

    @property
    def name(self):
        return "SafeCommute (INT8)" if self.quantized else "SafeCommute (ours)"

    def load(self, device='cpu'):
        self.device = device

        # Load feature stats
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH) as f:
                s = json.load(f)
            self.feat_mean = s['mean']
            self.feat_std = s['std']

        model = SafeCommuteCNN()
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        model.eval()

        if self.quantized:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.GRU}, dtype=torch.qint8)

        self.model = model.to(device)

    def predict_from_spectrogram(self, spec_tensor):
        """Predict from pre-computed, normalized spectrogram tensor (1, 1, 64, 188)."""
        with torch.no_grad():
            logits = self.model(spec_tensor.to(self.device))
            prob = torch.softmax(logits, dim=1)[0][1].item()
        return prob

    def predict_from_waveform(self, waveform):
        """Predict from raw waveform (numpy array, 16kHz)."""
        tensor = preprocess(waveform, self.feat_mean, self.feat_std)
        return self.predict_from_spectrogram(tensor)

    def get_model(self):
        return self.model
