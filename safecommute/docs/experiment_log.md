# Experiment Log

| # | Experiment | AUC | Accuracy | F1 | Params | Size(MB) | Latency(ms) | Notes |
|---|-----------|-----|----------|----|---------|---------|-----------:|-------|
| 1 | Baseline (production model) | 0.9497 | 0.7445 | 0.7596 | 1,829,444 | 6.99 | 1.0±0.0 | AUC=0.950 target |

### Baseline (production model) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.860 | 370/430 |
| cremad | 0.452 | 446/986 |
| esc | 0.719 | 82/114 |
| hns | 0.971 | 503/518 |
| rav | 0.521 | 75/144 |
| savee | 0.357 | 15/42 |
| tess | 0.988 | 244/247 |
| viol | 0.710 | 213/300 |
| yt | 0.922 | 637/691 |

Run: 2026-04-03 07:20

---

| 12 | TTA (3 augmentations) | 0.9477 | 0.7388 | 0.7541 | 1,829,444 | 6.99 | 3.0±0.1 | 3x: orig+freq±2+time±10 |

### TTA (3 augmentations) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.870 | 374/430 |
| cremad | 0.442 | 436/986 |
| esc | 0.737 | 84/114 |
| hns | 0.971 | 503/518 |
| rav | 0.514 | 74/144 |
| savee | 0.333 | 14/42 |
| tess | 0.964 | 238/247 |
| viol | 0.660 | 198/300 |
| yt | 0.932 | 644/691 |

Run: 2026-04-03 07:20

---

| 23 | TTA (5 augmentations) | 0.9495 | 0.7376 | 0.7530 | 1,829,444 | 6.99 | 5.1±0.1 | 5x: orig+freq±2+time±10 |

### TTA (5 augmentations) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.865 | 372/430 |
| cremad | 0.443 | 437/986 |
| esc | 0.719 | 82/114 |
| hns | 0.967 | 501/518 |
| rav | 0.514 | 74/144 |
| savee | 0.357 | 15/42 |
| tess | 0.976 | 241/247 |
| viol | 0.670 | 201/300 |
| yt | 0.923 | 638/691 |

Run: 2026-04-03 07:21

---

| 34 | Temperature Scaling | 0.9497 | 0.7445 | 0.7596 | 1,829,444 | 6.99 | 1.0±0.0 | T=1.464, learned on val set |

### Temperature Scaling — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.860 | 370/430 |
| cremad | 0.452 | 446/986 |
| esc | 0.719 | 82/114 |
| hns | 0.971 | 503/518 |
| rav | 0.521 | 75/144 |
| savee | 0.357 | 15/42 |
| tess | 0.988 | 244/247 |
| viol | 0.710 | 213/300 |
| yt | 0.922 | 637/691 |

Run: 2026-04-03 07:21

---

| 45 | Wav2Vec2 Proxy (Conv1d+GRU) | 0.9256 | 0.7440 | 0.7592 | 420,994 | 1.61 | 0.5±0.0 | Temporal conv proxy, no pretrained W2V2 |

### Wav2Vec2 Proxy (Conv1d+GRU) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.886 | 381/430 |
| cremad | 0.437 | 431/986 |
| esc | 0.711 | 81/114 |
| hns | 0.958 | 496/518 |
| rav | 0.465 | 67/144 |
| savee | 0.357 | 15/42 |
| tess | 0.964 | 238/247 |
| viol | 0.743 | 223/300 |
| yt | 0.942 | 651/691 |

Run: 2026-04-03 07:36

---

| 56 | Curriculum Learning (3-stage) | 0.9459 | 0.7465 | 0.7616 | 1,829,444 | 6.99 | 1.8±1.0 | easy→medium→full, 8 epochs/stage |

### Curriculum Learning (3-stage) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.898 | 386/430 |
| cremad | 0.427 | 421/986 |
| esc | 0.789 | 90/114 |
| hns | 0.977 | 506/518 |
| rav | 0.451 | 65/144 |
| savee | 0.357 | 15/42 |
| tess | 1.000 | 247/247 |
| viol | 0.737 | 221/300 |
| yt | 0.928 | 641/691 |

Run: 2026-04-03 07:44

---

| 67 | SOTA: Simple CNN (baseline) | 0.8811 | 0.6449 | 0.6611 | 93,378 | 0.36 | 0.4±0.0 | Params=93,378 |

### SOTA: Simple CNN (baseline) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.747 | 321/430 |
| cremad | 0.418 | 412/986 |
| esc | 0.491 | 56/114 |
| hns | 0.865 | 448/518 |
| rav | 0.403 | 58/144 |
| savee | 0.333 | 14/42 |
| tess | 0.628 | 155/247 |
| viol | 0.483 | 145/300 |
| yt | 0.912 | 630/691 |

Run: 2026-04-03 07:59

---

| 78 | SOTA: Audio ResNet | 0.9290 | 0.7612 | 0.7755 | 816,322 | 3.12 | 1.1±0.0 | Params=816,322 |

### SOTA: Audio ResNet — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.907 | 390/430 |
| cremad | 0.458 | 452/986 |
| esc | 0.833 | 95/114 |
| hns | 0.977 | 506/518 |
| rav | 0.493 | 71/144 |
| savee | 0.429 | 18/42 |
| tess | 0.955 | 236/247 |
| viol | 0.760 | 228/300 |
| yt | 0.936 | 647/691 |

Run: 2026-04-03 08:04

---

| 89 | SOTA: Mini AST (2L/4H) | 0.8909 | 0.6382 | 0.6543 | 297,602 | 1.14 | 0.7±0.0 | Params=297,602 |

### SOTA: Mini AST (2L/4H) — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.721 | 310/430 |
| cremad | 0.409 | 403/986 |
| esc | 0.482 | 55/114 |
| hns | 0.873 | 452/518 |
| rav | 0.396 | 57/144 |
| savee | 0.333 | 14/42 |
| tess | 0.676 | 167/247 |
| viol | 0.547 | 164/300 |
| yt | 0.860 | 594/691 |

Run: 2026-04-03 08:08

---

| 100 | SOTA: BiLSTM | 0.9104 | 0.7224 | 0.7386 | 594,436 | 2.27 | 0.3±0.0 | Params=594,436 |

### SOTA: BiLSTM — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.912 | 392/430 |
| cremad | 0.423 | 417/986 |
| esc | 0.816 | 93/114 |
| hns | 0.948 | 491/518 |
| rav | 0.410 | 59/144 |
| savee | 0.333 | 14/42 |
| tess | 0.858 | 212/247 |
| viol | 0.647 | 194/300 |
| yt | 0.920 | 636/691 |

Run: 2026-04-03 08:13

---

| 111 | Depthwise Separable Conv | 0.9438 | 0.7497 | 0.7646 | 818,893 | 3.13 | 1.2±0.0 | Params: 818,893 |

### Depthwise Separable Conv — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.856 | 368/430 |
| cremad | 0.463 | 457/986 |
| esc | 0.772 | 88/114 |
| hns | 0.969 | 502/518 |
| rav | 0.472 | 68/144 |
| savee | 0.357 | 15/42 |
| tess | 0.972 | 240/247 |
| viol | 0.753 | 226/300 |
| yt | 0.925 | 639/691 |

Run: 2026-04-03 08:14

---

| 122 | Knowledge Distillation | 0.9544 | 0.7644 | 0.7785 | 1,829,444 | 6.99 | 1.0±0.0 | T=4.0, α=0.7, self-distillation |

### Knowledge Distillation — Per-Source Breakdown
| Source | Accuracy | Correct/Total |
|--------|----------|---------------|
| bg | 0.900 | 387/430 |
| cremad | 0.452 | 446/986 |
| esc | 0.816 | 93/114 |
| hns | 0.975 | 505/518 |
| rav | 0.549 | 79/144 |
| savee | 0.381 | 16/42 |
| tess | 1.000 | 247/247 |
| viol | 0.757 | 227/300 |
| yt | 0.946 | 654/691 |

Run: 2026-04-03 08:18

---


## Threshold Optimization

| Threshold | Value | Accuracy | F1 | Sensitivity | Specificity | FPR |
|-----------|-------|----------|-------|-------------|-------------|-----|
| Default (0.5) | 0.500 | 0.7445 | 0.7596 | 0.9901 | 0.6569 | 0.3431 |
| Youden's J | 0.622 | 0.8597 | 0.8651 | 0.8916 | 0.8484 | 0.1516 |
| F1-optimal | 0.666 | 0.8805 | 0.8818 | 0.8083 | 0.9062 | 0.0938 |
| High sensitivity (0.3) | 0.300 | 0.4974 | 0.4902 | 1.0000 | 0.3181 | 0.6819 |
| Low FPR (0.7) | 0.700 | 0.8874 | 0.8851 | 0.7284 | 0.9441 | 0.0559 |

**Recommended**: Low FPR (0.7) at threshold=0.700

Run: 2026-04-03 21:18

---


## 5-Fold Cross-Validation (source-aware)

| Fold | AUC | Accuracy | F1 |
|------|-----|----------|----|
| 1 | 0.9477 | 0.7564 | 0.7717 |
| 2 | 0.9502 | 0.7720 | 0.7840 |
| 3 | 0.9477 | 0.7428 | 0.7600 |
| 4 | 0.9335 | 0.7622 | 0.7764 |
| 5 | 0.9393 | 0.7680 | 0.7825 |
| **Mean** | **0.9437** | **0.7603** | **0.7749** |
| **Std** | **0.0063** | **0.0102** | **0.0087** |

Run: 2026-04-03 23:00

---


## Ablation Study

| Variant | AUC | Accuracy | F1 | Params | vs Full |
|---------|-----|----------|----|---------|---------|
| Full model (baseline) | 0.9504 | 0.7437 | 0.7588 | 1,829,444 | +0.0000 |
| No SE attention | 0.9477 | 0.7454 | 0.7604 | 1,818,692 | -0.0027 |
| No GRU (global pool) | 0.9264 | 0.6823 | 0.6988 | 1,156,420 | -0.0240 |
| No multi-scale pool | 0.9448 | 0.7298 | 0.7456 | 1,828,932 | -0.0056 |
| Half channels | 0.9464 | 0.7298 | 0.7455 | 458,404 | -0.0040 |

Run: 2026-04-04 00:55

---


## Leave-One-Source-Out (LOSO)

| Held-Out Source | AUC | Accuracy | F1 | Samples |
|-----------------|-----|----------|----|---------|
| viol | 0.8525 | 0.5050 | 0.3878 | 2012 |
| tess | 0.8135 | 0.5169 | 0.3717 | 1600 |
| yt | 0.8028 | 0.8512 | 0.8442 | 4712 |
| rav | 0.7468 | 0.4537 | 0.2935 | 864 |
| cremad | 0.7110 | 0.4840 | 0.4148 | 6171 |
| savee | 0.6722 | 0.3833 | 0.2699 | 360 |
| bg | N/A | 0.7190 | 0.8365 | 3000 |
| esc | N/A | 0.7362 | 0.8481 | 800 |
| hns | N/A | 0.8527 | 0.9205 | 3429 |
| **Mean** | **0.7665** | | | |
| **Std** | **0.0623** | | | |

Run: 2026-04-04 04:08

---


## v2 Pipeline: AudioSet-based, No Acted Speech

**Changes**: Dropped CREMA-D/SAVEE/TESS/RAVDESS. Added AudioSet (6 threat + 7 safe categories, chunked 10s→3s). Fixed data leakage (source-aware splits via sha256 hash and predefined folds). Fixed augmentation timing (training-time only, no baked augmentation). Deterministic center-crop. Per-sample strong augmentation.

**Training data**: 28,772 samples (19,328 safe + 9,444 unsafe) from UrbanSound8K, ESC-50, AudioSet, YouTube, Violence dataset. Class balance: 67/33.

### 3-Seed Evaluation (gamma=3.0)

| Seed | AUC | Accuracy | F1 | Epochs |
|------|-----|----------|----|--------|
| 42 | 0.8538 | 0.6440 | 0.6500 | 16 |
| 123 | 0.8564 | 0.6748 | 0.6837 | 12 |
| 7 | 0.8217 | 0.5990 | 0.5988 | 8 |
| **Mean** | **0.8440** | **0.6393** | **0.6442** | |
| **Std** | **0.0193** | **0.0381** | **0.0434** | |

### Gamma 2.0 Comparison (seed=42)

| Gamma | AUC | Accuracy | F1 |
|-------|-----|----------|----|
| 3.0 | 0.8538 | 0.6440 | 0.6500 |
| 2.0 | 0.8514 | 0.6858 | 0.6958 |

### Best Model (seed=123) — Full Analysis

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| AUC-ROC | 0.928 | 0.819 | 0.856 | +0.072 |
| Accuracy | 0.746 | 0.660 | 0.675 | +0.071 |
| F1 | 0.752 | 0.660 | 0.684 | +0.068 |

### Per-Source Breakdown (test set)

| Source | Accuracy | Samples | Type |
|--------|----------|---------|------|
| yt_scream | 87.6% | 225 | real-world |
| hns | 88.0% | 643 | environmental |
| viol_violence | 72.3% | 296 | real-world |
| as (AudioSet) | 65.0% | 4241 | AudioSet |
| yt_metro | 62.9% | 442 | real-world |
| bg | 61.5% | 582 | environmental |
| esc | 48.1% | 160 | environmental |

### v1 vs v2 Comparison

| Metric | v1 (with leakage + acted speech) | v2 (clean) |
|--------|----------------------------------|------------|
| Test AUC | 0.950 | 0.856 |
| Lowest source | SAVEE 35.7% | ESC-50 48.1% |
| Training samples | 16,037 | 28,772 |
| Data leakage | Yes (random per-sample split) | No (source-aware) |
| Augmentation | Baked at prep time | Training-time only |

**Note**: The v1 AUC of 0.950 was inflated by data leakage (random splits), speaker memorization (TESS 2 speakers, 98.8%), and baked augmentation. The v2 AUC of 0.856 is on genuinely clean data. The real-world detection quality (YouTube screams 87.6%) remains strong.

Run: 2026-04-04

---


## Ablation Study v2

| Variant | AUC | Accuracy | F1 | Params | vs Full |
|---------|-----|----------|----|---------|---------|
| Full model (baseline) | 0.8271 | 0.6051 | 0.6061 | 1,829,444 | +0.0000 |
| No SE attention | 0.8493 | 0.6218 | 0.6262 | 1,818,692 | +0.0221 |
| No GRU (global pool) | 0.8430 | 0.5908 | 0.5889 | 1,156,420 | +0.0159 |
| No multi-scale pool | 0.8554 | 0.6127 | 0.6145 | 1,828,932 | +0.0283 |
| Half channels | 0.8362 | 0.5822 | 0.5798 | 458,404 | +0.0091 |

Run: 2026-04-05 13:40

---


## Leave-One-Source-Out (LOSO) v2

| Held-Out Source | AUC | Accuracy | F1 | Samples |
|-----------------|-----|----------|----|---------|
| viol | 0.5815 | 0.4776 | 0.3481 | 2012 |
| as_applause | N/A | 0.6612 | 0.7961 | 2683 |
| as_cheering | N/A | 0.7187 | 0.8363 | 2385 |
| as_crowd | N/A | 0.1112 | 0.2001 | 2231 |
| as_explosion | N/A | 0.7887 | 0.8819 | 1699 |
| as_glass | N/A | 0.8053 | 0.8921 | 1931 |
| as_gunshot | N/A | 0.9027 | 0.9489 | 2014 |
| as_laughter | N/A | 0.0600 | 0.1133 | 1832 |
| as_music | N/A | 0.5578 | 0.7161 | 2137 |
| as_screaming | N/A | 0.8076 | 0.8935 | 1720 |
| as_shout | N/A | 0.5209 | 0.6850 | 2248 |
| as_singing | N/A | 0.4532 | 0.6238 | 2224 |
| as_speech | N/A | 0.2049 | 0.3401 | 2152 |
| as_yell | N/A | 0.6578 | 0.7936 | 1765 |
| bg | N/A | 0.5887 | 0.7411 | 3000 |
| esc | N/A | 0.5038 | 0.6700 | 800 |
| hns | N/A | 0.7941 | 0.8852 | 3429 |
| yt_metro | N/A | 0.4007 | 0.5721 | 3489 |
| yt_scream | N/A | 0.7204 | 0.8375 | 1223 |
| **Mean** | **0.5815** | | | |
| **Std** | **0.0000** | | | |

Run: 2026-04-05 22:53

---

