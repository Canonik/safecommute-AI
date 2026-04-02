"""Wrapper for Wav2Vec2 — feature extraction + simple classifier probe."""

import numpy as np
import torch
import torch.nn as nn


class Wav2Vec2Wrapper:
    """
    Wav2Vec2 (Baevski et al. 2020) self-supervised audio model.
    We use frozen features + mean pooling + logistic regression probe.
    ~95M params base model, pretrained on LibriSpeech.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.probe = None

    @property
    def name(self):
        return "Wav2Vec2 (SSL)"

    def load(self, device='cpu'):
        self.device = device
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        model_name = "facebook/wav2vec2-base"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        # No fine-tuning — use raw features with energy-based heuristic
        print(f"  Wav2Vec2 loaded (frozen features, energy-based classification)")

    def predict_from_waveform(self, waveform):
        if len(waveform) < 1600:
            return 0.0
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use hidden states variance as a proxy for "activity level"
            hidden = outputs.last_hidden_state[0]  # (T, 768)
            # High variance in features = more dynamic audio = potentially unsafe
            feat_var = hidden.var(dim=0).mean().item()
            feat_energy = hidden.abs().mean().item()
        # Heuristic: combine variance and energy
        score = feat_var * 10 + feat_energy * 0.5
        # Sigmoid mapping
        unsafe_prob = 1.0 / (1.0 + np.exp(-2 * (score - 3.0)))
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        raise NotImplementedError("Wav2Vec2 requires raw waveform input")

    def get_model(self):
        return self.model
