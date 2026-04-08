# SafeCommute AI — Next Steps

## Completed

- [x] v2 Pipeline (clean data, source-aware splits, no leakage)
- [x] Drop acted speech datasets (CREMA-D, SAVEE, TESS, RAVDESS)
- [x] AudioSet strongly-labeled data (6 threat + 7 safe categories)
- [x] 3-seed evaluation, 5-fold CV, LOSO, ablation studies
- [x] SOTA benchmark (7 models)
- [x] Threshold optimization (Youden's J, F1-optimal, low-FPR)
- [x] Fine-tuning pipeline for deployment personalization
- [x] Deployment acceptance test suite (7 tests)
- [x] Model export (float32, INT8, TorchScript)
- [x] AST knowledge distillation (teacher AUC 0.85, student AUC 0.80)
- [x] 7-cycle experiment loop (gamma sweep, noise injection, etc.)
- [x] Discovery: gamma=3.0 over-regularization was root cause of hard negative failure
- [x] Best model: gamma=0.5 + noise injection (AUC 0.804, Acc 70.3%)

## In Progress

- [ ] Cycle 7: noise injection + label smoothing evaluation
- [ ] Consolidate experiment branches, merge best model

## Next: Fix Speech False Positive Problem (CRITICAL)

- [ ] Download high-quality speech data (LibriSpeech clean-100, CommonVoice verified, VoxCeleb1)
- [ ] Integrate 15-20k speech clips as safe class
- [ ] Retrain with gamma=0.5 + noise injection + speech data
- [ ] Speech FPR must drop below 30% (currently 72%)

## Next: Deployment Validation

- [ ] Create reliability-first eval protocol (deployment gates)
- [ ] Record real metro ambient audio (3-4 hours, phone in pocket)
- [ ] Fine-tune with recorded metro data
- [ ] Raspberry Pi ARM inference latency benchmark
- [ ] Field pilot: 1 week at metro station with logging

## Next: Paper

- [ ] Write paper (architecture, data strategy, experiment results, deployment)
- [ ] Re-run SOTA benchmark with fair fine-tuned baselines
- [ ] Generate final camera-ready figures
- [ ] Submit to DCASE Workshop / Interspeech / ICASSP
