"""Wrapper for PANNs CNN14 (SOTA AudioSet-pretrained model)."""

import numpy as np
import librosa


# AudioSet class names that map to "unsafe" (escalation/distress)
UNSAFE_LABELS = {
    'Screaming', 'Shout', 'Yell', 'Crying, sobbing',
    'Whimper', 'Groan', 'Battle cry',
    'Baby cry, infant cry', 'Children shouting',
}


class PANNsWrapper:
    """Wraps PANNs CNN14 for benchmarking against SafeCommute."""

    def __init__(self):
        self.model = None
        self.unsafe_indices = []

    @property
    def name(self):
        return "PANNs CNN14 (SOTA)"

    def load(self, device='cpu'):
        self.device = device
        try:
            from panns_inference import AudioTagging
            self.at = AudioTagging(checkpoint_path=None, device=device)
            # Find unsafe class indices
            for i, label in enumerate(self.at.labels):
                if label in UNSAFE_LABELS:
                    self.unsafe_indices.append(i)
            print(f"  PANNs loaded: {len(self.unsafe_indices)} unsafe AudioSet classes mapped")
        except ImportError:
            print("  Warning: panns_inference not installed. Install with: pip install panns_inference")
            raise

    def predict_from_waveform(self, waveform):
        """
        Predict from raw 16kHz waveform.
        PANNs expects 32kHz — we resample internally.
        """
        waveform_32k = librosa.resample(waveform, orig_sr=16000, target_sr=32000)
        audio_input = waveform_32k[np.newaxis, :]  # (1, samples)
        clipwise_output, _ = self.at.inference(audio_input)
        probs = clipwise_output[0]  # (527,) sigmoid probabilities

        # Max over unsafe classes
        unsafe_prob = max(probs[i] for i in self.unsafe_indices) if self.unsafe_indices else 0.0
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        raise NotImplementedError("PANNs requires raw waveform input")

    def get_model(self):
        return self.at.model if hasattr(self, 'at') else None
