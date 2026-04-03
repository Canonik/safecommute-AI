# SafeCommute AI — Next Steps for Publication & Deployment

## Current Status Summary

### What We Have
- **Production model**: CNN6+SE+GRU, AUC=0.950, 1.83M params, 7ms CPU inference
- **Clean dataset**: 16,037 train / 3,439 val / 3,472 test from 9 sources, only 35 flagged samples
- **No overfitting**: Train-test AUC gap = 0.028
- **Noise robust**: AUC >0.94 at σ=0.2
- **Comprehensive benchmarks**: 10+ experiments, 4 SOTA comparisons

### Key Weakness
- **Domain gap**: CREMA-D 45%, RAVDESS 52%, SAVEE 36% — acted speech ≠ real distress
- **Calibration**: ECE=0.268 — model overconfident, needs temperature scaling for deployment
- **High source variance**: std=0.221 across sources

---

## Phase 1: Anti-Overfitting Hardening (Priority: HIGH)

### 1.1 Cross-Validation Framework
- Implement **k-fold cross-validation** (k=5) with stratified source-aware splits
- Report mean ± std AUC across folds — more credible than single split
- Ensure no source leakage: same video's segments must be in same fold
- **Why**: Single train/val/test split can be lucky; cross-val proves robustness

### 1.2 Leave-One-Source-Out (LOSO) Evaluation  
- Train on 8 sources, test on the held-out 1 — repeat for all 9
- This directly measures generalization to unseen domains
- Expected: YouTube/HNS perform well (real-world), CREMA-D/SAVEE perform poorly (acted)
- **Why**: The ultimate overfitting test — if it generalizes to unseen sources, it's real

### 1.3 Regularization Sweep
- Test stronger dropout (0.3→0.5), weight decay (1e-4→1e-3), label smoothing (0.1→0.2)
- Stochastic depth (randomly drop entire ConvBlocks during training)
- Max norm constraint on weights
- **Why**: Ensure we're using optimal regularization strength

### 1.4 Early Stopping with Patience Analysis
- Plot learning curves for all experiments: train loss vs val loss over epochs
- Identify the exact epoch where overfitting begins
- Use this to set optimal patience and epoch count
- **Why**: Documents that we stop training at the right moment

---

## Phase 2: Dataset Cleanup & Formalization (Priority: HIGH)

### 2.1 Remove the 35 Flagged Samples
- Delete flat/uniform spectrograms identified in audit (all from YouTube)
- Verify no quality impact: retrain and compare
- Document removal justification in data report

### 2.2 Source Documentation
- For each of the 9 sources, document:
  - Original paper/URL, license, collection methodology
  - Sample count, recording conditions, label definitions
  - How labels were mapped to safe/unsafe
- Create `research/data_sources.md` with full provenance

### 2.3 Inter-Annotator Agreement (if applicable)
- For manually labeled data (YouTube), compute inter-rater reliability
- If only single-annotator: acknowledge as limitation
- **Why**: Academic papers require label quality documentation

### 2.4 Class Balance Analysis
- Current: 74% safe / 26% unsafe — document handling strategy
- Show that focal loss + class weights adequately address imbalance
- Experiment: undersample safe class to 50/50 — does it change results?

### 2.5 Train/Val/Test Split Verification
- Verify no data leakage between splits
- Ensure stratification by source AND class
- Document split strategy (random? stratified? temporal?)

---

## Phase 3: Make Experiments Academic-Grade (Priority: MEDIUM)

### 3.1 Statistical Significance
- Run each experiment 3-5 times with different seeds
- Report mean ± std for all metrics
- Use paired t-test or Wilcoxon signed-rank test vs baseline
- Mark results as significant (p<0.05) or not
- **Why**: Single runs are anecdotal; multiple runs with stats are evidence

### 3.2 Ablation Study
- Systematic ablation of each architectural component:
  1. Full model (baseline)
  2. Remove SE blocks
  3. Remove GRU (replace with global pooling)
  4. Remove multi-scale pooling (use only last hidden)
  5. Reduce channels (64→32, 128→64, 256→128)
- This proves each component contributes
- **Why**: Reviewers always ask "why this architecture?"

### 3.3 Confusion Matrix Analysis
- Per-source confusion matrices (not just accuracy)
- False positive analysis: what does the model wrongly flag as unsafe?
- False negative analysis: what unsafe events does the model miss?
- **Why**: In a safety system, false negatives are critical

### 3.4 Inference Profiling
- Detailed latency breakdown: feature extraction vs model forward pass
- Memory profiling: peak RAM during inference
- Compare: float32 vs int8 quantized vs ONNX
- Test on actual edge devices (Raspberry Pi, Jetson Nano)
- **Why**: "7ms on GPU" means nothing if you can't prove it runs on the target device

---

## Phase 4: Additional Experiments for Publication (Priority: MEDIUM)

### 4.1 Domain Adaptation Experiments
- MMD (Maximum Mean Discrepancy) loss between source domains
- Domain-adversarial training: learn source-invariant features
- This could fix the CREMA-D/RAVDESS gap
- **Why**: The domain gap is the paper's main challenge — showing we addressed it is key

### 4.2 Data Augmentation Ablation
- Current: SpecAugment + Gaussian noise + time shift + mixup
- Test each augmentation individually and in combination
- Find the minimal effective augmentation set
- **Why**: Proves each augmentation contributes, not just adds noise

### 4.3 Threshold Optimization
- Current: default 0.5 threshold
- Optimize threshold on validation set using Youden's J, F1-max, or cost-sensitive
- Show ROC and precision-recall curves with operating points
- **Why**: Deployment requires a well-justified threshold

### 4.4 Real-World Recording Test
- Record actual public transport audio (with consent)
- Test model on genuinely unseen real-world data
- Even 50-100 clips would be powerful evidence
- **Why**: The ultimate proof of generalization

---

## Phase 5: Publication Readiness (Priority: LOW until above done)

### 5.1 Paper Structure
1. Introduction: problem, motivation (GDPR, edge deployment)
2. Related Work: 20 papers from literature review
3. Dataset: 9 sources, cleaning process, distribution
4. Method: architecture, training recipe, anti-overfitting measures
5. Experiments: ablation, SOTA comparison, robustness
6. Results: per-source breakdown, cross-validation, domain analysis
7. Discussion: what worked, limitations, domain gap analysis
8. Conclusion: deployment-ready, future work

### 5.2 Code Cleanup
- Type annotations for all public functions
- Docstrings with parameter descriptions
- requirements.txt with pinned versions
- Reproducibility script: single command to reproduce all results
- **Do NOT over-engineer** — clean, readable code is better than abstracted code

### 5.3 Figures for Paper
- Architecture diagram (CNN6+SE+GRU flow)
- Per-source accuracy heatmap
- ROC curves (all experiments overlaid)
- Confusion matrix
- Noise robustness degradation curve
- Training curves (loss over epochs)

---

## Immediate Action Items (This Week)

1. [ ] Remove 35 flagged samples, retrain baseline, verify no regression
2. [ ] Run 5-fold cross-validation with source-aware stratification
3. [ ] Run leave-one-source-out evaluation
4. [ ] Run 3x repeat of top experiments for statistical significance
5. [ ] Create source documentation (`data_sources.md`)
6. [ ] Run ablation study (remove SE, remove GRU, reduce channels)
7. [ ] Optimize classification threshold on validation set
8. [ ] Apply temperature scaling (T=1.46) and re-evaluate calibration
9. [ ] Generate all publication figures

---

## What NOT to Do

- **Don't chase higher AUC** — 0.950 on a clean test set is solid. Focus on proving robustness.
- **Don't add more data sources** without careful quality control — each new source adds domain complexity.
- **Don't change the architecture** before proving the current one is well-understood (ablation).
- **Don't publish** before cross-validation results are in — single-split results are weak evidence.
- **Don't use CREMA-D/SAVEE accuracy** as a failure metric — these are fundamentally different from real distress, and correctly classifying them is not the goal.
