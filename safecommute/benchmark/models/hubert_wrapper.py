"""Wrapper for HuBERT — self-supervised audio model."""

import numpy as np
import torch


class HuBERTWrapper:
    """
    HuBERT (Hsu et al. 2021) self-supervised audio model.
    Similar to Wav2Vec2 but uses offline clustering for targets.
    ~95M params, pretrained on LibriSpeech.
    Uses frozen features with energy-based heuristic (no fine-tuning).
    """

    def __init__(self):
        self.model = None
        self.processor = None

    @property
    def name(self):
        return "HuBERT (SSL)"

    def load(self, device='cpu'):
        self.device = device
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        model_name = "facebook/hubert-base-ls960"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"  HuBERT loaded (frozen features, energy-based classification)")

    def predict_from_waveform(self, waveform):
        if len(waveform) < 1600:
            return 0.0
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[0]  # (T, 768)
            feat_var = hidden.var(dim=0).mean().item()
            feat_energy = hidden.abs().mean().item()
        score = feat_var * 10 + feat_energy * 0.5
        unsafe_prob = 1.0 / (1.0 + np.exp(-2 * (score - 3.0)))
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        raise NotImplementedError("HuBERT requires raw waveform input")

    def get_model(self):
        return self.model
