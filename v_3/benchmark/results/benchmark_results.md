# SafeCommute AI — Benchmark Results

**Date**: 2026-04-03 00:13

**Test set**: 3472 spectrogram samples (2559 safe, 913 unsafe)

**Waveform test**: 1419 raw audio samples

## Comparison Table

| Model                      |   Params | Size(MB) | Lat.(ms) |    Acc |     F1 |    AUC |
|----------------------------|----------|----------|----------|--------|--------|--------|
| SafeCommute (ours)         |    1.83M |      7.0 |      7.4 |  0.745 |  0.760 |  0.950 |
| SafeCommute (INT8)         |    1.15M |      5.0 |      6.6 |  0.743 |  0.758 |  0.950 |
| Energy Baseline            |        0 |      0.0 |      0.0 |  0.263 |  0.110 |  0.503 |
| PANNs CNN14                |    81.8M |    320.0 |    250.0 |  0.579 |  0.425 |  0.660 |

## Key Advantages of SafeCommute

- **AUC-ROC: 0.950** — strong discrimination between safe and unsafe audio
- **7.0 MB** on disk — 45x smaller than PANNs CNN14 (320 MB)
- **7.4 ms** inference — real-time on CPU, suitable for Raspberry Pi
- **GDPR compliant** — no raw audio storage, only non-reconstructible spectrograms
- **Domain-specific** — fine-tuned for escalation detection, not general audio
