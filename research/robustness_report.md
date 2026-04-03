# Robustness Evaluation Report

Generated: 2026-04-03 07:50

**Goal:** Verify the model generalizes well and doesn't just overfit training data.

## 1. Train-Test Gap (Overfitting Indicator)

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| AUC | 0.9783 | 0.9589 | 0.9497 | +0.0286 (OK) |
| ACCURACY | 0.7709 | 0.7682 | 0.7445 | +0.0264 (OK) |
| F1 | 0.7849 | 0.7822 | 0.7596 | +0.0253 (OK) |

**Verdict: NO overfitting detected.** Train-test AUC gap is minimal.

## 2. Noise Robustness

Adding Gaussian noise to spectrograms at test time:

| Noise σ | AUC | Accuracy | AUC Drop |
|---------|-----|----------|----------|
| 0.05 | 0.9496 | 0.7445 | +0.0001 |
| 0.10 | 0.9488 | 0.7503 | +0.0009 |
| 0.20 | 0.9443 | 0.7558 | +0.0054 |
| 0.50 | 0.8717 | 0.8295 | +0.0780 |
| 1.00 | 0.5728 | 0.7258 | +0.3769 |

*A robust model should degrade gracefully with noise.*

## 3. Time Masking Robustness

Masking portions of the spectrogram (simulating partial/corrupted audio):

| Mask Ratio | AUC | Accuracy | AUC Drop |
|------------|-----|----------|----------|
| 10% | 0.9472 | 0.7359 | +0.0026 |
| 20% | 0.9452 | 0.7451 | +0.0045 |
| 30% | 0.9367 | 0.7601 | +0.0130 |
| 50% | 0.8654 | 0.7716 | +0.0843 |

*Moderate drops under masking are expected; large drops suggest fragile features.*

## 4. Confidence Calibration

- **Brier Score:** 0.1662 (lower is better, <0.25 is good)
- **Expected Calibration Error (ECE):** 0.2678 (lower is better, <0.1 is good)

**Poorly calibrated.** Temperature scaling strongly recommended.

## 5. Cross-Source Generalization

- **Mean accuracy across sources:** 0.722
- **Std across sources:** 0.221
- **Range:** 0.357 — 0.988 (spread = 0.631)

| Source | Accuracy | vs Mean |
|--------|----------|---------|
| savee | 0.357 | -0.365 |
| cremad | 0.452 | -0.270 |
| rav | 0.521 | -0.201 |
| viol | 0.710 | -0.012 |
| esc | 0.719 | -0.003 |
| bg | 0.860 | +0.138 |
| yt | 0.922 | +0.200 |
| hns | 0.971 | +0.249 |
| tess | 0.988 | +0.266 |

**High variance.** Model performs very differently across sources.

## 6. Deployment Readiness Summary

| Check | Status | Detail |
|-------|--------|--------|
| Overfitting | PASS | AUC gap: 0.029 |
| Noise robustness | PASS | AUC@σ=0.2: 0.944 |
| Calibration | FAIL | ECE: 0.268 |
| Source generalization | FAIL | Std: 0.221 |

**Score: 2/4 checks passed.**
