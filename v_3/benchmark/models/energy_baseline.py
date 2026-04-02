"""RMS energy baseline — classifies based on audio loudness alone."""

import numpy as np


class EnergyBaseline:
    """
    Simplest possible classifier: if RMS energy > threshold, predict unsafe.
    Answers the question: "Does our model beat raw loudness detection?"
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    @property
    def name(self):
        return "Energy RMS Baseline"

    def load(self, device='cpu'):
        pass  # no model to load

    def predict_from_waveform(self, waveform):
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        # Sigmoid-like mapping for probability
        unsafe_prob = 1.0 / (1.0 + np.exp(-30 * (rms - self.threshold)))
        return float(unsafe_prob)

    def predict_from_spectrogram(self, spec_tensor):
        # Approximate: use spectrogram energy as proxy
        energy = float(spec_tensor.mean())
        unsafe_prob = 1.0 / (1.0 + np.exp(-0.5 * (energy + 10)))
        return float(unsafe_prob)

    def calibrate(self, waveforms, labels):
        """Find optimal RMS threshold on validation data."""
        best_f1, best_thresh = 0, self.threshold
        for t in np.arange(0.01, 0.2, 0.005):
            preds = [(np.sqrt(np.mean(w**2)) > t) for w in waveforms]
            tp = sum(1 for p, l in zip(preds, labels) if p and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p and l == 0)
            fn = sum(1 for p, l in zip(preds, labels) if not p and l == 1)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        self.threshold = best_thresh
        print(f"  Energy baseline calibrated: threshold={best_thresh:.3f}, F1={best_f1:.3f}")

    def get_model(self):
        return None
