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

