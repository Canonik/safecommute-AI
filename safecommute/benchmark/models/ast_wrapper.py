"""Wrapper for Audio Spectrogram Transformer (AST) — HuggingFace."""

import numpy as np
import torch
import librosa


class ASTWrapper:
    """
    Audio Spectrogram Transformer (Gong et al. 2021).
    Pretrained on AudioSet via HuggingFace transformers.
    ~87M params, SOTA on ESC-50 (95.6%) and AudioSet (0.485 mAP).
    """

    UNSAFE_LABELS = {
        'Screaming', 'Shout', 'Yell', 'Crying, sobbing',
        'Whimper', 'Groan', 'Battle cry',
        'Baby cry, infant cry', 'Children shouting',
    }

    def __init__(self):
        self.model = None
        self.processor = None
        self.unsafe_indices = []

    @property
    def name(self):
        return "AST (Transformer)"

    def load(self, device='cpu'):
        self.device = device
        from transformers import ASTForAudioClassification, ASTFeatureExtractor
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.processor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        # Map AudioSet labels to unsafe
        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            if label in self.UNSAFE_LABELS:
                self.unsafe_indices.append(int(idx))
        print(f"  AST loaded: {len(self.unsafe_indices)} unsafe classes mapped")

    def predict_from_waveform(self, waveform):
        # AST expects 16kHz
        if len(waveform) < 1600:
            return 0.0
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        unsafe_prob = max((probs[i] for i in self.unsafe_indices), default=0.0)
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        raise NotImplementedError("AST requires raw waveform input")

    def get_model(self):
        return self.model
