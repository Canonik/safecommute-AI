# SafeCommute AI — Next Steps

## Completed (v2 Pipeline)

- [x] Drop acted speech datasets (CREMA-D, SAVEE, TESS, RAVDESS)
- [x] Add AudioSet strongly-labeled data (6 threat + 7 safe categories)
- [x] Fix data leakage (source-aware splits via sha256 hash + predefined folds)
- [x] Fix augmentation timing (training-time only, no baked augmentation)
- [x] 3-seed evaluation with confidence intervals
- [x] LOSO evaluation (19 sources)
- [x] Ablation study (5 variants)
- [x] SOTA benchmark (7 models including PANNs, AST, Wav2Vec2)
- [x] Fine-tuning script for deployment personalization
- [x] Deployment acceptance test suite (7 tests)
- [x] Threshold optimization (Youden's J, F1-optimal, low-FPR)
- [x] Silence handling (synthetic samples + energy gating)
- [x] Model export (float32, INT8, TorchScript)
- [x] HuggingFace model upload (Canonik/SafeCommute)

## Next: Paper Writing

- [ ] Write paper (architecture, data strategy, results, deployment)
- [ ] Generate final camera-ready figures
- [ ] Submit to target venue

## Next: Deployment

- [ ] Record real metro ambient audio (3-4 hours, phone in pocket)
- [ ] Fine-tune with recorded metro data (expect FP rate < 5%)
- [ ] Record bar/nightlife ambient for second vertical
- [ ] Raspberry Pi deployment test (ARM inference latency)
- [ ] Field pilot: 1 week at metro station with logging

## Future Work

- [ ] Environmental noise mixing at training time (SNR augmentation)
- [ ] Multi-language support (Italian, other languages)
- [ ] Multi-microphone beamforming for direction estimation
- [ ] Streaming inference optimization (overlap-add, buffer management)
- [ ] Federated learning for privacy-preserving model updates
