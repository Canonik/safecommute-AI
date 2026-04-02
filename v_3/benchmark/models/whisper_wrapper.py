"""Wrapper for OpenAI Whisper — used as audio feature extractor."""

import numpy as np
import torch


class WhisperWrapper:
    """
    OpenAI Whisper (Radford et al. 2023) audio model.
    Trained on 680k hours of web audio. Primarily ASR but
    the encoder provides strong general audio features.
    ~39M params (tiny), ~74M (base).
    Uses encoder features with energy heuristic.
    """

    def __init__(self, size="tiny"):
        self.size = size
        self.model = None
        self.processor = None

    @property
    def name(self):
        return f"Whisper-{self.size}"

    def load(self, device='cpu'):
        self.device = device
        from transformers import WhisperModel, WhisperFeatureExtractor
        model_name = f"openai/whisper-{self.size}"
        self.processor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"  Whisper-{self.size} loaded (encoder features, energy-based)")

    def predict_from_waveform(self, waveform):
        if len(waveform) < 1600:
            return 0.0
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Use only encoder
            encoder_out = self.model.encoder(**inputs)
            hidden = encoder_out.last_hidden_state[0]  # (T, D)
            feat_var = hidden.var(dim=0).mean().item()
            feat_energy = hidden.abs().mean().item()
        score = feat_var * 8 + feat_energy * 0.3
        unsafe_prob = 1.0 / (1.0 + np.exp(-2.5 * (score - 2.5)))
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        raise NotImplementedError("Whisper requires raw waveform input")

    def get_model(self):
        return self.model
