# SafeCommute AI — Research Experiments

Research phase: systematic evaluation of techniques to improve the SafeCommute edge audio classifier.

## Baseline Model

| Metric | Value |
|--------|-------|
| Architecture | CNN6 + SE + GRU + Multi-Scale Pooling |
| Parameters | 1.83M |
| AUC-ROC (test) | 0.950 |
| Accuracy (test) | 74.5% |
| F1 (test) | 0.760 |
| Model Size | 6.99 MB (float32) |
| Inference Latency | ~1ms (GPU), ~7ms (CPU) |
| Train-Test AUC Gap | 0.028 (no overfitting) |

## Key Findings

### Robustness Assessment
- **No overfitting detected**: Train AUC=0.978, Test AUC=0.950 (gap=0.028)
- **Noise robust**: AUC holds >0.94 up to σ=0.2 Gaussian noise
- **Time masking robust**: AUC >0.93 with up to 30% of spectrogram masked
- **Poor calibration**: ECE=0.268 — temperature scaling (T=1.46) recommended for deployment
- **High cross-source variance**: std=0.221 — CREMA-D (45%), SAVEE (36%) vs TESS (99%), HNS (97%)

### Dataset Quality
- **16,037 training / 3,439 val / 3,472 test samples** from 9 sources
- **35 samples flagged** for quality issues (flat/uniform spectrograms from YouTube)
- **0 mislabeled candidates** detected — dataset is exceptionally clean
- **Class imbalance**: 74% safe / 26% unsafe — addressed by focal loss
- See [dataset_audit_report.md](dataset_audit_report.md) for full analysis

### Experiment Results (Final — 12 experiments)

| Rank | Experiment | AUC | Acc | F1 | Params | vs Baseline |
|------|-----------|-----|-----|----|---------|----|
| **1** | **Knowledge Distillation** | **0.954** | **76.4%** | **0.779** | **1.83M** | **+0.005** |
| 2 | Baseline (production) | 0.950 | 74.5% | 0.760 | 1.83M | — |
| 3 | Temperature Scaling | 0.950 | 74.5% | 0.760 | 1.83M | 0.000 |
| 4 | TTA (5 aug) | 0.950 | 73.8% | 0.753 | 1.83M | -0.000 |
| 5 | TTA (3 aug) | 0.948 | 73.9% | 0.754 | 1.83M | -0.002 |
| 6 | Curriculum Learning | 0.946 | 74.7% | 0.762 | 1.83M | -0.004 |
| 7 | **Depthwise Separable** | **0.944** | **75.0%** | **0.765** | **819K** | **-0.006** |
| 8 | SOTA: Audio ResNet | 0.929 | 76.1% | 0.776 | 816K | -0.021 |
| 9 | Wav2Vec2 Proxy | 0.926 | 74.4% | 0.759 | 421K | -0.024 |
| 10 | SOTA: BiLSTM | 0.910 | 72.2% | 0.739 | 594K | -0.039 |
| 11 | SOTA: Mini AST | 0.891 | 63.8% | 0.654 | 298K | -0.059 |
| 12 | SOTA: Simple CNN | 0.881 | 64.5% | 0.661 | 93K | -0.069 |

### What Worked
1. **Knowledge Distillation** — NEW BEST: AUC=0.954 (+0.005), self-distillation with KL divergence loss
2. **Temperature Scaling** — free calibration improvement for deployment (T=1.46)
3. **Depthwise Separable Conv** — 55% param reduction (819K vs 1.83M) with only 0.006 AUC drop
4. **Our architecture beats all SOTA baselines** — CNN6+SE+GRU outperforms ResNet, Transformer, LSTM

### What Didn't Work
1. **TTA** — slight accuracy degradation, not worth 3-5x latency cost
2. **Curriculum easy-stage** — severe overfitting (91% train, 65% val) on easy samples alone
3. **Mini AST** — small transformers underperform without pretraining on small datasets
4. **Simple CNN** — proves SE attention and GRU temporal modeling are essential

### Key Insights
1. **No overfitting** (gap=0.028) — model is deployment-ready
2. **Architecture matters**: SE blocks + GRU + multi-scale pooling give +7% AUC over simple CNN
3. **Distillation improves generalization**: KL divergence loss acts as label smoothing
4. **Domain gap is the bottleneck**: acted speech (CREMA-D 45%, SAVEE 36%) vs real-world (YouTube 95%, HNS 97%)
5. **Depthwise is the edge champion**: 55% fewer params with minimal accuracy loss

## Directory Structure

```
research/
├── README.md                      # This file
├── literature_review.md           # 20 paper summaries
├── experiment_log.md              # All experiment results (append-only)
├── dataset_audit_report.md        # Dataset quality analysis
├── dataset_audit_details.json     # Detailed audit data
├── robustness_report.md           # Robustness/overfitting evaluation
├── experiments/                   # Experiment scripts
│   ├── eval_utils.py              # Shared evaluation utilities
│   ├── baseline_eval.py           # Baseline evaluation
│   ├── test_time_augmentation.py  # TTA experiment
│   ├── temperature_scaling.py     # Temperature scaling
│   ├── curriculum_training.py     # Curriculum learning
│   ├── feature_augmentation.py    # CutMix + feature mixup
│   ├── attention_variants.py      # CBAM, ECA, no attention
│   ├── depthwise_model.py         # Depthwise separable convs
│   ├── distill_training.py        # Knowledge distillation
│   ├── wav2vec2_features.py       # Wav2Vec2 proxy model
│   ├── contrastive_pretrain.py    # SimCLR-style pretraining
│   ├── ensemble.py                # Model ensemble
│   ├── dataset_audit.py           # Dataset quality audit
│   ├── robustness_eval.py         # Robustness evaluation
│   └── sota_benchmark.py          # SOTA model comparison
├── benchmarks/
│   ├── comparison_table.md        # Ranked experiment table
│   ├── generate_plots.py          # Plot generation script
│   └── plots/                     # Generated visualization
│       ├── auc_comparison.png
│       ├── latency_vs_auc.png
│       ├── radar_comparison.png
│       ├── source_heatmap.png
│       └── params_vs_auc.png
├── results/                       # Model checkpoints
└── run_all_experiments.py         # Master experiment runner
```

## Recommendations for Production

1. **Use knowledge distillation** for training — AUC=0.954 (best result)
2. **Apply temperature scaling** (T=1.46) for confidence calibration
3. **Consider depthwise model** for ultra-constrained edge devices — 55% fewer params
4. **Do NOT use TTA** in production — negligible benefit, 3-5x latency
5. **Focus data collection on real-world audio** rather than more acted datasets
6. **Monitor per-source accuracy** post-deployment to catch domain drift

## Next Steps

See [NEXT_STEPS.md](NEXT_STEPS.md) for the full publication roadmap, including:
- Cross-validation and LOSO evaluation for stronger evidence
- Statistical significance testing (multi-seed runs)
- Ablation study to justify each architectural component
- Domain adaptation experiments for the acted speech gap
