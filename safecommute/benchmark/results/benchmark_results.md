# SafeCommute AI — Benchmark Results

**Date**: 2026-04-05 (v2 test set)

**Test set**: 6,589 spectrogram samples (4,541 safe, 2,048 unsafe)

## Comparison Table

| Model | Params | Size(MB) | Lat.(ms) | Acc | F1 | AUC |
|-------|--------|----------|----------|-----|----|----|
| SafeCommute (ours) | 1.83M | 7.0 | 12.0 | 0.675 | 0.684 | 0.856 |
| SafeCommute (INT8) | 1.15M | 5.0 | 10.5 | 0.675 | 0.684 | 0.856 |
| Energy Baseline | 0 | 0.0 | 0.0 | 0.311 | 0.147 | 0.451 |
| PANNs CNN14 | 81.8M | 320.0 | 250.0 | 0.593 | 0.468 | 0.624 |
| AST (Transformer) | 86.6M | 330.3 | 964.8 | 0.602 | 0.469 | 0.615 |
| Wav2Vec2 (SSL) | 94.4M | 360.0 | 191.6 | 0.606 | 0.458 | 0.523 |
| Whisper-tiny | 37.8M | 144.0 | 132.1 | 0.394 | 0.222 | 0.580 |

**Note**: PANNs, AST, Wav2Vec2, and Whisper were evaluated out-of-the-box on their 527/1000+ AudioSet classes with manual unsafe class aggregation. They were NOT fine-tuned on our binary task. A fair comparison would fine-tune each on our data — this is planned for the paper.

## Key Advantages

- **45x smaller** than PANNs CNN14 (7MB vs 320MB)
- **21x faster** than PANNs CNN14 (12ms vs 250ms)
- **Privacy-preserving** — no raw audio storage, only PCEN spectrograms
- **Domain-specific** — fine-tuned for escalation detection

## Updated Best Config (April 2026)

After the experiment cycle loop, the best training config is gamma=0.5 focal loss + noise injection (AUC 0.804 on updated test set). The benchmark above uses the original v2 gamma=3.0 model — results should be re-run with the current best config for the paper.
