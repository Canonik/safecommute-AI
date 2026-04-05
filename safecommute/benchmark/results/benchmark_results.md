# SafeCommute AI — Benchmark Results

**Date**: 2026-04-05 11:54

**Test set**: 6589 spectrogram samples (4541 safe, 2048 unsafe)

**Waveform test**: 5204 raw audio samples

## Comparison Table

| Model                      |   Params | Size(MB) | Lat.(ms) |    Acc |     F1 |    AUC |
|----------------------------|----------|----------|----------|--------|--------|--------|
| SafeCommute (ours)         |    1.83M |      7.0 |     12.0 |  0.675 |  0.684 |  0.856 |
| SafeCommute (INT8)         |    1.15M |      5.0 |     10.5 |  0.675 |  0.684 |  0.856 |
| Energy Baseline            |        0 |      0.0 |      0.0 |  0.311 |  0.147 |  0.451 |
| PANNs CNN14                |    81.8M |    320.0 |    250.0 |  0.593 |  0.468 |  0.624 |
| AST (Transformer)          |    86.6M |    330.3 |    964.8 |  0.602 |  0.469 |  0.615 |
| Wav2Vec2 (SSL)             |    94.4M |    360.0 |    191.6 |  0.606 |  0.458 |  0.523 |
| Whisper-tiny               |    37.8M |    144.0 |    132.1 |  0.394 |  0.222 |  0.580 |

## Key Advantages of SafeCommute

- **AUC-ROC: 0.856** — strong discrimination between safe and unsafe audio
- **7.0 MB** on disk — 45x smaller than PANNs CNN14 (320 MB)
- **12.0 ms** inference — real-time on CPU, suitable for Raspberry Pi
- **GDPR compliant** — no raw audio storage, only non-reconstructible spectrograms
- **Domain-specific** — fine-tuned for escalation detection, not general audio
