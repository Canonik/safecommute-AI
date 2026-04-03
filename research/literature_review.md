# Literature Review - SafeCommute AI

> Compiled 2026-04-03. Papers 1-5 and 6-10 are sourced from training knowledge (web search was unavailable to those agents). Papers 11-15 and 16-20 are from agent outputs also based on training knowledge. All citations should be verified against original sources before use in publications.

---

## Category 1: Audio Event Detection & Scream Detection (Papers 1-5)

*Agent 1 searched: "scream detection deep learning 2024", "audio aggression detection CNN", "sound event detection edge deployment", "abnormal sound detection public space", "acoustic surveillance violence detection". WebSearch permission was denied. Summaries below are from training knowledge.*

### Paper 1: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (PANNs)

- **Authors/Year/Venue:** Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., Plumbley, M. D. / 2020 / IEEE/ACM Transactions on Audio, Speech, and Language Processing
- **Key technique:** A family of pretrained CNN architectures (CNN6, CNN10, CNN14) trained on the full AudioSet (5800+ hours, 527 classes) using log-mel spectrograms, establishing strong baselines for general-purpose audio pattern recognition.
- **Dataset & Results:** AudioSet (2M clips). CNN14 achieves mAP=0.431 on AudioSet (SOTA at publication). Transfer learning to ESC-50 yields 94.7% accuracy. CNN6 (the smallest) achieves mAP=0.343 with only 4.7M params.
- **Applicable to SafeCommute:** SafeCommute's CNN6+SE+GRU architecture is already inspired by PANNs. Specific idea: initialize the CNN6 backbone from PANNs pretrained weights (available on GitHub) rather than random init, then fine-tune. This should dramatically improve CREMA-D performance since the pretrained features are more general. Load weights with `model.load_state_dict(panns_weights, strict=False)` matching only the conv layers.
- **Implementation effort:** Easy -- download pretrained weights, write a weight-loading function (~30 lines), adjust layer names to match SafeCommuteCNN's conv blocks.

### Paper 2: Scream and Gunshot Detection and Localization for Audio-Surveillance Systems

- **Authors/Year/Venue:** Valenzise, G., Gerosa, L., Tagliasacchi, M., Antonacci, F., Sarti, A. / 2007 / IEEE Conference on Advanced Video and Signal Based Surveillance (AVSS)
- **Key technique:** Two-stage pipeline using MFCC features with GMM classifiers for scream/gunshot detection, followed by acoustic source localization using microphone arrays. Foundational work in acoustic surveillance.
- **Dataset & Results:** Custom dataset of screams, gunshots, and background noise in indoor environments. Detection accuracy ~90% for screams at 10dB SNR, degrading to ~70% at 0dB SNR. False alarm rate <5% in controlled settings.
- **Applicable to SafeCommute:** The SNR-dependent performance analysis is directly relevant. SafeCommute should evaluate AUC at different SNR levels (mix unsafe audio with bus ambient noise at -5, 0, 5, 10 dB). This would reveal if the model's real-world failure modes are SNR-related. Add a `test_snr_robustness()` function to `comprehensive_analysis.py`.
- **Implementation effort:** Easy -- mix test samples with noise at various SNR levels and re-evaluate, ~50 lines of code.

### Paper 3: Sound Event Detection of Weakly Labelled Data with CNN-Transformer and Automatic Threshold Optimization

- **Authors/Year/Venue:** Miyazaki, K., Komatsu, T., Hayashi, T., Watanabe, S., Toda, T., Takeda, K. / 2020 / DCASE 2020 Workshop
- **Key technique:** CNN encoder with Transformer decoder for sound event detection, trained with weak labels (clip-level) using attention pooling. Automatic threshold optimization via validation F1 maximization replaces manual threshold tuning.
- **Dataset & Results:** DCASE 2020 Task 4 (weakly labeled AudioSet subset, 10 classes). Event-based F1 of 44.3%, segment-based F1 of 71.2%, outperforming the CRNN baseline by 6+ points.
- **Applicable to SafeCommute:** The automatic threshold optimization is immediately useful. SafeCommute currently uses a fixed 0.5 threshold for safe/unsafe. Implement threshold search on validation set: sweep thresholds from 0.1-0.9 in steps of 0.01, pick the one maximizing F1 (or a custom metric weighting recall higher for unsafe class). This alone could improve real-world performance by 2-5%.
- **Implementation effort:** Easy -- add `optimize_threshold()` to the evaluation pipeline, ~20 lines.

### Paper 4: Audio Spectrogram Transformer (AST)

- **Authors/Year/Venue:** Gong, Y., Chung, Y.-A., Glass, J. / 2021 / Interspeech 2021
- **Key technique:** Applies a pure Vision Transformer (ViT) to audio spectrograms by splitting the mel spectrogram into 16x16 patches, using positional embeddings, and fine-tuning from ImageNet-pretrained ViT weights. No convolution layers at all.
- **Dataset & Results:** AudioSet mAP=0.459 (SOTA), ESC-50 accuracy=95.6%. However, AST-base has 87M parameters and requires ~30ms inference on GPU.
- **Applicable to SafeCommute:** AST is too large for edge deployment (87M vs SafeCommute's 1.83M), but the patch-based spectrogram tokenization idea can be adapted. Specific idea: replace the first conv layer with a patch embedding layer (16x16 patches projected to 64-dim), feed patches into the existing GRU as a sequence. This gives the GRU better temporal resolution without increasing params much. Alternatively, use AST as a teacher for knowledge distillation into SafeCommuteCNN.
- **Implementation effort:** Hard (architecture change) or Medium (knowledge distillation).

### Paper 5: FSD50K: An Open Dataset of Human-Labeled Sound Events

- **Authors/Year/Venue:** Fonseca, E., Favory, X., Pons, J., Font, F., Serra, X. / 2022 / IEEE/ACM Transactions on Audio, Speech, and Language Processing
- **Key technique:** A large-scale, open, human-validated dataset for sound event recognition with 51,197 clips covering 200 classes from the AudioSet ontology, with quality-controlled annotations via Freesound Annotator.
- **Dataset & Results:** 51K clips, 200 classes, 108.3 hours. Baseline CNN14 achieves mAP=0.558. Includes classes directly relevant to SafeCommute: "Screaming", "Shout", "Glass breaking", "Siren", "Crying".
- **Applicable to SafeCommute:** Add FSD50K as a 9th data source. Filter for relevant unsafe classes (Screaming, Shout, Crying, Glass breaking) and safe classes (Speech, Music, Silence, Vehicle noise). This gives high-quality human-validated samples that bridge the acted-vs-real gap, since FSD50K sources are from Freesound (real recordings, not acted). Add to `download_datasets.py`.
- **Implementation effort:** Medium -- download FSD50K (~5GB), filter relevant classes, extract mel spectrograms, integrate into the data pipeline.

---

## Category 2: Efficient/Lightweight Audio Models for Edge Deployment (Papers 6-10)

*Agent 2 searched: "lightweight audio classification edge device", "tiny audio neural network", "efficient spectrogram CNN mobile", "audio model compression quantization pruning", "sub-1MB audio classifier embedded". WebSearch permission was denied. Summaries below are from training knowledge.*

### Paper 6: EfficientAT: Efficient Audio Transformers

- **Authors/Year/Venue:** Schmid, F., Masoudian, S., Koutini, K., Widmer, G. / 2023 / Interspeech 2023
- **Key technique:** Applies MobileNetV3-style inverted bottleneck blocks with squeeze-and-excitation to audio spectrograms, combined with efficient multi-head attention in later stages. Uses progressive downsampling and knowledge distillation from a large teacher.
- **Dataset & Results:** AudioSet mAP=0.471 with only 6M params (MN-10 variant). Smallest variant (MN-01) achieves mAP=0.376 with 0.5M params and ~1ms inference on CPU. Outperforms PANNs CNN14 (13.7M params) while being 20x smaller.
- **Applicable to SafeCommute:** The MN-01 architecture (0.5M params, mAP=0.376) could replace SafeCommuteCNN (1.83M params) entirely, giving 3.6x parameter reduction with competitive performance. Alternatively, adopt only the inverted bottleneck blocks (expand-depthwise-project pattern) in the existing conv blocks to reduce FLOPs without changing the overall architecture.
- **Implementation effort:** Medium -- either replace the model architecture entirely or refactor ConvBlocks to use depthwise separable convolutions (~100 lines of model changes).

### Paper 7: BC-ResNet: Broadcasted Residual Learning for Efficient Keyword Spotting

- **Authors/Year/Venue:** Kim, B., Chang, S., Lee, J., Sung, D. / 2021 / Interspeech 2021
- **Key technique:** Frequency-wise broadcasting in residual connections -- instead of standard residual additions, broadcasts a 1D frequency-pooled feature across the time dimension, reducing temporal redundancy. Combined with sub-spectral normalization that normalizes sub-bands independently.
- **Dataset & Results:** Google Speech Commands V2: 98.0% accuracy with only 321K parameters and 17.5M MACs. Achieves similar accuracy to models 10x larger (ResNet-15 at 98.0% with 3.2M params).
- **Applicable to SafeCommute:** Two directly applicable ideas: (1) Sub-spectral normalization: split the 64-mel input into 4 sub-bands of 16 mels each, apply separate BatchNorm to each sub-band. This helps the model treat low-frequency (speech fundamentals) and high-frequency (screams, glass breaking) regions differently. (2) Frequency broadcasting in residual connections to reduce temporal FLOPs. Both can be added to existing ConvBlocks.
- **Implementation effort:** Easy (sub-spectral norm, ~20 lines) to Medium (full BC-ResNet architecture swap).

### Paper 8: Keyword Transformer: A Self-Attention Model for Keyword Spotting

- **Authors/Year/Venue:** Berg, A., O'Connor, M., Cruz, M. T. / 2021 / Interspeech 2021
- **Key technique:** Replaces the temporal modeling stage (typically GRU/LSTM) with a multi-head self-attention Transformer operating on frame-level CNN features. Uses a lightweight CNN frontend for local feature extraction followed by a Transformer encoder for global temporal context.
- **Dataset & Results:** Speech Commands V2: 98.6% accuracy with 607K params. Outperforms both pure CNN and pure Transformer baselines, showing the CNN+Transformer hybrid is optimal for small audio models.
- **Applicable to SafeCommute:** Replace the GRU layer in SafeCommuteCNN with a 2-layer Transformer encoder (2 heads, 128-dim). Self-attention captures long-range temporal dependencies better than GRU for events like escalating arguments where the temporal pattern matters. The CNN frontend remains identical. Since SafeCommute already uses multi-scale pooling after temporal modeling, the Transformer output feeds directly into the existing pooling layer.
- **Implementation effort:** Medium -- replace GRU with `nn.TransformerEncoder` (~30 lines), adjust dimensions, retrain. May need learning rate tuning.

### Paper 9: TinyML for Audio: Deploying Neural Networks on Microcontrollers

- **Authors/Year/Venue:** Banbury, C., Zhou, C., Fedorov, I., et al. (MLPerf Tiny benchmark team) / 2021 / NeurIPS 2021 Benchmarks Track
- **Key technique:** Systematic benchmark of audio ML models on microcontrollers (Cortex-M4/M7), covering quantization (int8/int4), pruning (structured/unstructured), and neural architecture search (NAS) for keyword spotting under strict memory constraints (<256KB SRAM, <1MB flash).
- **Dataset & Results:** Speech Commands V2 on Cortex-M4: 90.0% accuracy at 64KB model size (int8 DS-CNN), vs 93.5% at 500KB. Demonstrates that int8 quantization loses <1% accuracy while halving model size and inference time.
- **Applicable to SafeCommute:** SafeCommute is already 7MB float32, 5MB INT8. Next step: structured pruning (remove entire filters with <threshold L1 norm) followed by INT8 quantization could reach ~2MB with <1% AUC loss. Use PyTorch's `torch.nn.utils.prune.ln_structured` on conv layers, targeting 50% sparsity, then quantize with `torch.quantization.quantize_dynamic`.
- **Implementation effort:** Easy -- PyTorch has built-in pruning and quantization APIs. ~40 lines for a prune-and-quantize script.

### Paper 10: Knowledge Distillation in Audio Classification

- **Authors/Year/Venue:** Gong, Y., Lai, C.-I., Chung, Y.-A., Glass, J. / 2022 / IEEE ICASSP 2022
- **Key technique:** Distills knowledge from large AST teacher (87M params) into small CNN student using soft label distillation (KL divergence on temperature-scaled logits) plus intermediate feature matching between teacher and student hidden layers.
- **Dataset & Results:** Student (CNN6-like, ~5M params) achieves mAP=0.416 on AudioSet when distilled from AST teacher, vs mAP=0.343 without distillation -- a 21% relative improvement. The distilled student nearly matches the teacher at 17x fewer params.
- **Applicable to SafeCommute:** Use a pretrained PANNs CNN14 or AST as teacher. Train SafeCommuteCNN with combined loss: `L = alpha * focal_loss(student_pred, label) + (1-alpha) * KL(student_logits/T, teacher_logits/T)` with T=4, alpha=0.5. The teacher provides richer supervision than binary labels, particularly for ambiguous samples near the decision boundary. This should improve CREMA-D performance since the teacher has seen much more diverse audio.
- **Implementation effort:** Medium -- requires downloading a pretrained teacher, running inference on all training data to cache soft labels (or run teacher in parallel during training), and modifying the loss function.

---

## Category 3: Domain Adaptation for Audio Classification (Papers 11-15)

*Agent 3 provided these from training knowledge. SafeCommute context: weak on acted speech (CREMA-D 45.2%, RAVDESS 52%) but strong on real-world (YouTube screams 97.3%). The acted-to-real gap is the key challenge.*

### Paper 11: Unsupervised Domain Adaptation for Speech Emotion Recognition Using PCANet

- **Authors/Year/Venue:** Zheng, W., Yu, J., Zou, Y. / 2015 / Multimedia Tools and Applications
- **Key technique:** Uses PCANet with Transfer Component Analysis (TCA) to align feature distributions between source (acted) and target (real) emotion corpora by minimizing Maximum Mean Discrepancy (MMD) in a shared subspace.
- **Dataset & Results:** Cross-corpus evaluation on EmoDB to CASIA and vice versa. Achieved ~10-15% absolute improvement over no-adaptation baselines, reaching ~55-65% on cross-corpus emotion recognition.
- **Applicable to SafeCommute:** Add an MMD loss term between CREMA-D/RAVDESS mel-spectrogram features and YouTube real-world features in the GRU bottleneck layer. During training, minimize classification loss on labeled acted data + MMD loss to align acted and real feature distributions. This directly attacks the acted-to-real gap without needing labels on real data.
- **Implementation effort:** Easy -- MMD is ~20 lines of PyTorch code added to the training loop. Use a Gaussian kernel, compute MMD between mini-batch embeddings from each domain after the GRU layer.

### Paper 12: Deep Domain Adaptation for Sound Event Detection

- **Authors/Year/Venue:** Wei, Y., Xu, Q., Kong, Q., Plumbley, M. D. / 2020 / ICASSP 2020
- **Key technique:** Domain adversarial training (gradient reversal layer) applied to sound event detection CNNs -- a domain discriminator tries to distinguish source/target features while the feature extractor learns domain-invariant representations via gradient reversal.
- **Dataset & Results:** Adapted between synthetic Scaper-generated soundscapes and real DCASE recordings. Showed 5-12% F1 improvement on target domain for sound event detection tasks.
- **Applicable to SafeCommute:** Add a domain discriminator head (2-layer MLP) after the SE+GRU block in `SafeCommuteCNN`. During training, feed acted (CREMA-D/RAVDESS) and real (YouTube) spectrograms, use a Gradient Reversal Layer (GRL) so the CNN backbone learns features that fool the domain discriminator while still classifying safe/unsafe correctly. Tag each sample with a domain label (acted=0, real=1) in `TensorAudioDataset`.
- **Implementation effort:** Medium -- requires implementing GRL (a custom `torch.autograd.Function`, ~15 lines), adding a discriminator head, modifying the training loop.

### Paper 13: Cross-Corpus Speech Emotion Recognition Based on Domain-Adaptive Least-Squares Regression

- **Authors/Year/Venue:** Song, P., Zheng, W., Yu, J., Ou, S. / 2016 / IEEE Signal Processing Letters
- **Key technique:** Domain-adaptive least-squares regression that learns a shared projection where labeled source (acted) and unlabeled target (different corpus) data are mapped into a common emotion space, using structural risk minimization and distribution alignment.
- **Dataset & Results:** Cross-corpus experiments across Berlin EMO-DB, CASIA, SAVEE, and eNTERFACE. Achieved 45 to 58% on the hardest cross-corpus pairs (~+13% absolute).
- **Applicable to SafeCommute:** Freeze the CNN backbone trained on acted data, extract embeddings, then train a small regression/classifier head with domain regularization. Practically: freeze `SafeCommuteCNN` up to the GRU output, extract 128-dim embeddings for all data, train a 2-layer MLP with domain regularization. This is a fast experiment that does not require retraining the full model.
- **Implementation effort:** Easy -- freeze model, extract embeddings, train a small head. Can be done in a single script.

### Paper 14: Prototypical Networks for Few-Shot Audio Classification

- **Authors/Year/Venue:** Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., Salamon, J. / 2020 / ICASSP 2020
- **Key technique:** Episodic meta-learning with prototypical networks -- learns an embedding space where class prototypes (mean embeddings of support examples) enable classification with as few as 1-5 examples per class via nearest-centroid.
- **Dataset & Results:** ESC-50 and Speech Commands with 1-shot and 5-shot settings. 5-shot accuracy reached ~80% on unseen sound classes, significantly outperforming fine-tuning baselines.
- **Applicable to SafeCommute:** Use prototypical network-style evaluation as a calibration step at deployment. Collect 5-10 real-world "unsafe" clips from the target environment (e.g., a specific bus route), compute prototype embeddings, then classify new audio by distance to safe/unsafe prototypes. Export the GRU bottleneck features, compute class centroids from a few real examples, use cosine distance for inference. This sidesteps retraining entirely and is especially useful for adapting to new transit environments.
- **Implementation effort:** Easy -- no retraining required. Write an inference wrapper (~50 lines) that computes prototype embeddings from a calibration set and classifies by nearest centroid.

### Paper 15: Semi-Supervised Activity Recognition via Shared Feature Extractors With Domain Adaptation (Mean Teacher variant)

- **Authors/Year/Venue:** Sanabria, A. R. et al. / 2021 / IEEE Transactions on Multimedia (audio variant: Takahashi et al., DCASE 2018 Workshop)
- **Key technique:** Combines Mean Teacher semi-supervised learning with domain adversarial training -- a student model trains on labeled source + unlabeled target data, while a teacher (EMA of student weights) provides pseudo-labels on target data, with consistency regularization.
- **Dataset & Results:** DCASE workshop results show 4-5 point F1 improvement from consistency regularization on weakly-labeled sound event detection tasks.
- **Applicable to SafeCommute:** Maintain an EMA copy of `SafeCommuteCNN`, feed unlabeled real-world audio through both student (with augmentation) and teacher (without), minimize MSE between their predictions as a consistency loss. Combined with existing focal loss on labeled data. Update rule: `teacher_params = 0.999 * teacher_params + 0.001 * student_params`. This lets SafeCommute exploit large amounts of unlabeled real transit audio.
- **Implementation effort:** Medium -- requires maintaining a second model copy (EMA), modifying the training loop, and collecting/preparing unlabeled real transit audio.

---

## Category 4: Training Techniques & Loss Functions (Papers 16-20)

*Agent 4 provided these from training knowledge. Focuses on training strategies to improve SafeCommute's robustness and handle class imbalance.*

### Paper 16: Focal Loss for Dense Object Detection (adapted to audio)

- **Authors/Year/Venue:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. / 2017 / ICCV 2017 (arXiv:1708.02002)
- **Key technique:** Introduces focal loss `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)` which down-weights easy negatives and focuses training on hard, misclassified examples.
- **Dataset & Results:** COCO object detection (original); in audio adaptations (DCASE challenges), focal loss with gamma=2-3 typically yields 2-5% F1 improvement on imbalanced sound event detection over cross-entropy.
- **Applicable to SafeCommute:** SafeCommute already uses focal loss with gamma=3. The key insight: alpha (per-class weighting) matters as much as gamma. Compute alpha dynamically per epoch as `alpha_t = 1 - (class_count / total_count)` rather than a fixed value. Since SafeCommute's safe/unsafe ratio shifts across the 8 data sources, dynamic alpha could improve CREMA-D performance (45.2%) by preventing the model from ignoring hard acted-speech samples.
- **Implementation effort:** Easy -- single-line change in `FocalLoss.forward()` to accept per-epoch alpha recomputed from current batch distribution.

### Paper 17: COLA: Contrastive Learning of General-Purpose Audio Representations

- **Authors/Year/Venue:** Saeed, A., Grangier, D., Zeghidour, N. / 2021 / ICASSP 2021 (arXiv:2010.10915)
- **Key technique:** Self-supervised contrastive pretraining on unlabeled audio: two augmented views of the same spectrogram segment are pulled together in embedding space while different segments are pushed apart, using bilinear similarity + NT-Xent loss.
- **Dataset & Results:** Pretrained on AudioSet (~5000 hours). ESC-50: 85.4% linear eval, 91.1% fine-tuned. Speech Commands V2: 96.5%. Outperforms supervised baselines when labeled data <1000 samples.
- **Applicable to SafeCommute:** Pretrain the CNN6+SE encoder contrastively on large unlabeled urban-transport audio (e.g., freesound.org bus/train ambience, 50+ hours). Add a projection head (2-layer MLP, 128-d) after the last ConvBlock, pretrain with NT-Xent using two SpecAugment views of each clip, then discard projection head and attach existing GRU + classifier. Should help CREMA-D weakness since encoder learns more general features before seeing labels.
- **Implementation effort:** Hard -- requires collecting unlabeled audio, implementing contrastive dataloader with dual-augmentation, NT-Xent loss, and a two-phase training script.

### Paper 18: SSAST: Self-Supervised Audio Spectrogram Transformer

- **Authors/Year/Venue:** Gong, Y., Lai, C.-I. J., Chung, Y.-A., Glass, J. / 2022 / AAAI 2022 (arXiv:2110.09784)
- **Key technique:** Masked spectrogram patch modeling (joint discriminative + generative pretraining) for Audio Spectrogram Transformers. Randomly masks patches and trains to reconstruct and discriminate real vs. fake patches.
- **Dataset & Results:** Pretrained on AudioSet + LibriSpeech. ESC-50: 88.8% (no external labels). Key finding: with only 10% of AudioSet labels, SSAST matches fully supervised AST performance.
- **Applicable to SafeCommute:** The masked patch prediction idea works on CNNs too. Implement masked spectrogram pretraining: randomly zero out 40-60% of mel spectrogram patches (8x8 blocks on the 64x188 input), train the CNN encoder to predict masked patch statistics (mean/std) via an auxiliary MSE head. Do this on existing 16K samples (ignoring labels), then fine-tune with labels. Forces the encoder to learn spectral structure rather than memorizing source-specific shortcuts.
- **Implementation effort:** Medium -- requires a patch masking function, small decoder head for reconstruction, and pretraining loop, but no new data or architecture changes to the core CNN.

### Paper 19: Curriculum Learning for Speech Emotion Recognition

- **Authors/Year/Venue:** Zhao, R., Li, H., Ma, T., Lu, K. / 2021 / Interspeech 2021
- **Key technique:** Orders training samples from easy to hard based on difficulty score (prediction confidence or loss magnitude), then feeds them in increasingly difficult batches following an anti-annealing curve.
- **Dataset & Results:** IEMOCAP 4-class emotion recognition: curriculum learning improves unweighted accuracy from 64.2% to 67.8% (+3.6 points), with biggest gains on confusion-prone pairs. On CREMA-D, improvements of 2-4% over random-order training.
- **Applicable to SafeCommute:** Directly addresses CREMA-D weakness (45.2%). (1) Train one epoch normally, record per-sample loss. (2) Sort samples by loss. (3) In subsequent epochs, use a pacing function: epoch `e` uses the easiest `min(1, e/E_pace)` fraction of data, where `E_pace` is ~60% of total epochs. Implement as a custom `CurriculumSampler` in the DataLoader. Acted-speech samples that confuse the model (hard) are introduced gradually.
- **Implementation effort:** Easy -- implement a custom `CurriculumSampler` (~40 lines) that wraps the existing `TensorAudioDataset`, reorder indices each epoch by cached loss values.

### Paper 20: Mean Teacher for Sound Event Detection (DCASE 2018 Task 4, 1st Place)

- **Authors/Year/Venue:** Lu, JiaKai / 2018 / DCASE 2018 Workshop
- **Key technique:** Mean Teacher semi-supervised learning with EMA of model weights as "teacher" generating pseudo-labels on unlabeled data. Student trained on both labeled data (standard loss) and unlabeled data (consistency loss between student and teacher predictions). Combined with SpecAugment and mixup.
- **Dataset & Results:** DCASE 2018 Task 4 (weakly labeled + unlabeled AudioSet, 10 classes). Event-based F1 of 32.4% (1st place), +10 point improvement over baseline. Mean teacher consistency loss accounted for ~4-5 points of the gain.
- **Applicable to SafeCommute:** (1) Duplicate `SafeCommuteCNN` as teacher (EMA decay=0.999). (2) For each batch, run labeled samples through student with focal loss as usual. (3) Run unlabeled transport audio through both student and teacher; add MSE consistency loss between softmax outputs, weighted by ramp-up coefficient `w(t) = exp(-5*(1 - t/T)^2)`. (4) Update teacher: `theta_teacher = 0.999 * theta_teacher + 0.001 * theta_student`. Exploits hundreds of hours of unlabeled bus/train audio without annotation cost.
- **Implementation effort:** Medium -- requires unlabeled data pipeline, EMA weight update logic, consistency loss, and ramp-up schedule, but model architecture stays identical.

---

## Summary: Recommended Implementation Priority

| Priority | Paper | Technique | Expected Gain | Effort |
|----------|-------|-----------|---------------|--------|
| 1 | 3 | Automatic threshold optimization | +2-5% F1 | Easy |
| 2 | 16 | Dynamic focal loss alpha | +1-3% on hard samples | Easy |
| 3 | 19 | Curriculum learning | +2-4% on CREMA-D | Easy |
| 4 | 11 | MMD domain alignment loss | +5-10% cross-domain | Easy |
| 5 | 14 | Prototypical calibration at deploy | Adaptive to new environments | Easy |
| 6 | 1 | PANNs pretrained weights init | +5-10% overall | Easy |
| 7 | 9 | Structured pruning + INT8 | 7MB to ~2MB model | Easy |
| 8 | 7 | Sub-spectral normalization | +1-2% on diverse audio | Easy |
| 9 | 12 | DANN gradient reversal | +5-12% cross-domain | Medium |
| 10 | 18 | Masked spectrogram pretraining | Better feature quality | Medium |
| 11 | 20 | Mean Teacher semi-supervised | Leverage unlabeled data | Medium |
| 12 | 10 | Knowledge distillation from AST | +10-20% relative | Medium |
| 13 | 6 | EfficientAT architecture | 3.6x param reduction | Medium |
| 14 | 8 | Replace GRU with Transformer | Better temporal modeling | Medium |
| 15 | 17 | Contrastive pretraining (COLA) | General features | Hard |

> Note: All papers cited from training knowledge should be verified against original sources (Google Scholar, Semantic Scholar, arXiv) before inclusion in any publication. Some specific numbers may be approximate.
