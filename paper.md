# SafeCommute AI — Paper + Product Plan (updated 2026-04-21)

Single source of truth for:
- what the model *actually does* right now (measured, not marketing),
- what is publishable at a workshop and what is not,
- what the public site claims and why each claim needs editing,
- what to do next, ordered by workshop-publishability leverage,
- pricing reality given measured accuracy.

Everything numeric below traces to one of three machine-readable files (regenerate before citing):

- [`tests/reports/verify_performance_claims.json`](tests/reports/verify_performance_claims.json) — Phase A, latency, size, INT8 parity
- [`tests/reports/phase_b_metro.json`](tests/reports/phase_b_metro.json) — per-site fine-tune + held-out measurement
- [`tests/reports/tweak_finetune.json`](tests/reports/tweak_finetune.json) — tweak sweep (architecture-preserving levers)

If a doc number disagrees with these JSONs, the JSONs win. Run `PYTHONPATH=. python tests/verify_performance_claims.py && PYTHONPATH=. python tests/finalize_release.py` to refresh.

---

## 0. TL;DR

**What works (reproducible from on-disk checkpoint `models/safecommute_v2.pth`):**

| Claim | Measured | Verdict |
|---|---|---|
| AUC-ROC 0.804 | 0.8044 | PASS |
| Accuracy @ 0.50 = 70.3% | 70.3% | PASS |
| F1 weighted @ 0.50 = 0.716 | 0.716 | PASS |
| Params 1.83M / size 7 MB FP32 | 1,829,444 / 7.00 MB | PASS |
| Per-source TPR (yell/scream/shout/gunshot/…) | Within ±3% of table | PASS |
| Speech FP ≈ 72% | 71.7% | PASS |
| INT8 ONNX size ≤ 6 MB | 3.72 MB | PASS |
| INT8 AUC degradation ≤ 0.02 | |Δ| = 0.002 | PASS |
| γ=0.5 is the stable operating point for this architecture | Validated | PASS |
| ONNX FP32 @ 8T Ryzen 7: 4.0 ms median | Measured 2026-04-21 | PASS (not Ryzen-5) |
| ONNX INT8 @ 8T Ryzen 7: 2.8 ms median | Measured 2026-04-21 | PASS |

**What does NOT reproduce (documented failure modes, the honest contributions):**

| Claim | Measured | Status |
|---|---|---|
| "41% leakage" (pitch figures) | 34.2% measured FPR@0.5 | Hand-typed literal, now removed |
| Post-fine-tune overall FP ≤ 5% (metro default recipe, held-out) | **38.2 %** | FAIL |
| Post-fine-tune overall FP ≤ 5% (metro, best of 4 tweaks) | **29.4 %** | FAIL — plateau, −8.8 pts from default but 5× the gate |
| Post-fine-tune speech FP ≤ 4.2% / 10% (default) | **15.1 %** | FAIL |
| Post-fine-tune recall ≈ 86% (default) | 89.5% | Passes ≥86% but outside ±3% abs band |
| "12 ms on CPU" (Ryzen 5, 1T) | Not reproducible (hw not on hand); Ryzen 7 1T = 26.9 ms | Historical only |
| γ=3.0 AUC 0.856 / 0% hard-neg | Checkpoint not preserved | Historical snapshot |
| n≥3 sites for Phase B | n=1 (metro) | Scope limitation |

**Tweak-sweep summary (5 architecture-preserving levers, all measured on 34 truly-held-out metro wavs):**

| Tweak | Recipe | Default-thr FP / recall | Best FP≤5% operating point |
|---|---|---|---|
| 0 | keep_safe 0.5, 10 ep, frozen | 38.2 % / 89.5 % | thr 0.77 → 2.9 % / 40.4 % |
| 1 | **keep_safe 0.2, 15 ep, frozen** | **29.4 %** / 89.5 % | thr 0.73 → 0.0 % / 35.1 % |
| 2 | keep_safe 0.3, 15 ep, **unfreeze + warmup 3** | 38.2 % / 87.7 % | thr 0.74 → 2.9 % / 43.9 % |
| 3 | keep_safe 0.1, 20 ep, frozen | **29.4 %** / 87.7 % | thr 0.71 → 2.9 % / **49.1 %** |
| 4 | threshold-only recal on default | — | thr 0.77 → 2.9 % / 40.4 % (= tweak 0) |

- **FP plateau at ~29.4 %** regardless of keep_safe ∈ {0.1, 0.2}. More-metro-weight *hits a ceiling*.
- **Unfreezing the CNN is actively worse** (tweak 2 regresses to the default 38.2 %).
- **Best FP≤5% operating point: tweak 3, recall 49.1 %** (+8.7 pts absolute vs default's 40.4 % at the same FP). The best *training-side* architecture-preserving lever is "metro dominates the safe class, CNN stays frozen."
- **The ≤ 5 % FP / ≥ 86 % recall gate is unreachable via training-side tweaks alone** with the 58-clip metro fine-tune pool. Closing it requires *post-hoc* interventions not yet implemented — see §7.

**Headline conclusion**: this is a real research artifact with workshop-paper-worthy findings (focal-γ collapse + precisely-quantified deployment-gap on a real per-site fine-tune with an architectural-preserving tweak sweep), wrapped in marketing claims that the measurement contradicts. Paper submission requires **(a)** the architecture untouched, **(b)** the doc numbers updated to the measured ones, **(c)** at least a token effort at n≥2 sites, **(d)** implementing site-ambient threshold recal + temporal-majority aggregation to see if *those* close the gap, and **(e)** a proper baseline vs YAMNet or PANNs CNN14 on the same test set under the same fine-tune protocol.

---

## 1. Measured numbers (authoritative)

### 1.1 Base model on `prepared_data/test/` (12,178 samples, 15% sha256 test split)

All from `tests/reports/verify_performance_claims.json` (run 2026-04-21).

- AUC-ROC = **0.804** (PASS vs claim 0.804 ± 0.010)
- Accuracy @ thr = 0.50 = **0.703** (PASS)
- F1-weighted @ thr = 0.50 = **0.716** (PASS)
- Micro threat recall = **0.809** (claim 0.82 ± 0.03 PASS; earlier "82%" was slightly optimistic)
- Overall FPR @ thr = 0.50 = **0.342** — this is the "leakage" number. The pitch deck had "41%" which was a hand-typed literal in [scripts/generate_pitch_figures.py:267-268](scripts/generate_pitch_figures.py); now measured.

**Per-source TPR (threat side, sorted desc):**

| Source | TPR | n |
|---|---|---|
| viol_violence | 0.963 | 133 |
| as_yell | 0.906 | 286 |
| as_gunshot | 0.895 | 342 |
| as_glass | 0.799 | 299 |
| as_screaming | 0.791 | 235 |
| yt_scream | 0.782 | 225 |
| as_explosion | 0.738 | 219 |
| as_shout | 0.647 | 309 |

**Per-source FPR (safe side, sorted desc — top is worst):**

| Source | FPR | n |
|---|---|---|
| as_laughter | **0.825** | 269 |
| as_speech | **0.717** | 378 |
| esc_hns | 0.625 | 40 |
| viol_violence (safe-side) | 0.607 | 163 |
| as_crowd | 0.579 | 423 |
| as_music | 0.376 | 484 |
| esc | 0.358 | 120 |
| yt_metro | 0.351 | 442 |
| as_singing | 0.269 | 353 |
| bg | 0.213 | 582 |
| hns | 0.138 | 643 |
| as_cheering | 0.126 | 380 |
| as_applause | 0.117 | 264 |
| synth_* | 0.000 | 225 |

Note: laughter FPR (82.5%) is *worse* than speech (71.7%). The prior narrative highlighted only speech; laughter is the bigger false-alarm driver.

### 1.2 Latency (hardware-disclosed)

From [tests/baseline/measure_latency_baseline.txt](tests/baseline/measure_latency_baseline.txt) and the `latency_and_size` rows of the verifier.

| Variant | Median / p99 (ms) | Threads | Hardware |
|---|---|---|---|
| Eager PyTorch + inference_mode | 8.5 / 9.1 | 8 | Ryzen 7 7435HS |
| Eager PyTorch + inference_mode | 34.2 / — | 1 | Ryzen 7 7435HS |
| ONNX FP32 | **4.0 / 4.2** | 8 | Ryzen 7 7435HS |
| ONNX FP32 | 26.9 / — | 1 | Ryzen 7 7435HS |
| ONNX INT8 static (Step 5 output) | **2.8 / 3.2** | 8 | Ryzen 7 7435HS |
| Preprocess only (librosa PCEN) | 16.0 / 30.0 | 8 | Ryzen 7 7435HS |
| End-to-end (preprocess + INT8) | 49.3 / 74.0 | 8 | Ryzen 7 7435HS |
| "~12 ms on Ryzen 5, 1T" | N/A | 1 | Hardware not on hand — **historical** |

Disclosure: `torch==2.11.0+cu130`, `onnxruntime==1.24.4`, governor=performance, BLAS=mkl, `python==3.14.4`. Every row re-runnable with `python tests/measure_latency.py` or `python tests/measure_e2e_latency.py`.

### 1.3 Per-site deployment (Phase B) — n=1 site (metro)

From `tests/reports/phase_b_metro.json`.

- Fine-tune set: **58 WAVs** from `raw_data/youtube_metro/`
- Held-out set: **34 WAVs** from `raw_data/youtube_metro_quarantine/` (by construction never seen by fine-tune)
- Threat set: 57 WAVs from `raw_data/youtube_screams/`
- Calibrated `low_fpr` threshold = 0.667 (chosen by fine-tune on universal test set)
- Determinism check: PASS
- Silence-gate check: PASS

**Deployment gate (from [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py)):**

| Gate | Target | Measured | Pass? |
|---|---|---|---|
| FP on held-out ambient | ≤ 5 % | **38.2 %** (13 / 34 files) | **FAIL** |
| Threat recall | ≥ 90 % | **89.5 %** (51 / 57 files) | near-miss |
| Speech FP post-FT (universal as_speech subset) | ≤ 10 % | **15.1 %** | FAIL |
| Latency | ≤ 15 ms mean | 4.0 ms ONNX | PASS |

**Threshold landscape on held-out (all re-runnable from `tests/tweak_finetune.py` → `tweak0`):**

| Threshold | FP rate | Threat recall |
|---|---|---|
| 0.500 | 91.2 % | 100.0 % |
| 0.600 | 76.5 % | 93.0 % |
| 0.667 *(default)* | 38.2 % | 89.5 % |
| 0.700 | 20.6 % | 82.5 % |
| 0.750 | 8.8 % | 54.4 % |
| **0.770** | **2.9 %** | **40.4 %** ← "best" FP≤5% point |
| 0.800 | 0.0 % | 22.8 % |

No operating point reaches both FP ≤ 5 % and recall ≥ 86 %. The ambient and threat max-prob distributions overlap structurally (ambient median 0.637, threat median 0.757; ambient max 0.786 overlaps threat min 0.533). This is *the* central honest finding.

### 1.4 Tweak sweep — architecture-preserving levers

From `tests/reports/tweak_finetune.json` (populated by `tests/run_tweaks_2_3.py`; **may still be running at time of writing — refresh the JSON before citing final numbers in the paper**).

1. **Tweak 0 — threshold recalibration on held-out** (no retrain).
   Best FP≤5%: thr=0.77 → FP 2.9 %, recall 40.4 %. *Not enough.*
2. **Tweak 1 — `--keep-safe-ratio 0.2 --epochs 15 --freeze-cnn`** (more metro in safe class).
   Training outcome: AUC 0.8031 (−0.0013), acc 0.7292 (+0.0258), F1 0.7380 (+0.0223). `low_fpr` recalibrated to 0.614.
   **Held-out (34 quarantine wavs) at thr=0.614**: FP = **29.4 %**, threat recall = **89.5 %**. Improvement over tweak 0 (default): FP drops 38.2 % → 29.4 % (−8.8 pts absolute), recall unchanged.
   **Threshold recal sweep**: smallest thr with FP ≤ 5 % on held-out is **0.730**, which gives FP = **0.0 %** but recall collapses to **35.1 %**. Same structural overlap as tweak 0 — no operating point reaches both FP ≤ 5 % AND recall ≥ 86 %.
   *Direction is right; magnitude still short of the gate.*
3. **Tweak 2 — `--warmup-epochs 3 --keep-safe-ratio 0.3 --epochs 15` (unfreeze CNN after 3-epoch frozen warmup)**.
   Training outcome: `low_fpr` recalibrated to 0.636 (between default and tweak 1).
   **Held-out at thr=0.636**: FP = **38.2 %**, threat recall = **87.7 %**. *No improvement over default (tweak 0)*; recall slightly lower (89.5 % → 87.7 %).
   **Threshold recal on held-out**: thr=0.740 gives FP = 2.9 %, recall = 43.9 % — essentially the same structural ceiling as tweak 0's (FP 2.9 % @ recall 40.4 %). Unfreezing did not shift the overall ROC shape.
   **Important finding: unfreezing the CNN does not help on this data.** The adaptation problem is not in the feature extractor — it is in the decision surface, which the frozen-CNN + more-metro-weight recipe (tweak 1) actually handles better. This is a concrete, reportable architectural observation: *the base CNN filters generalize to metro ambient; what doesn't generalize is the GRU + FC's decision function, and you fix that by re-weighting the training distribution, not by unfreezing*.
4. **Tweak 3 — `--keep-safe-ratio 0.1 --epochs 20 --freeze-cnn`** (metro dominates safe class).
   Training outcome: AUC 0.7955 (−0.0089 — largest drop), acc 0.7426 (+0.0392 — largest gain), F1 0.7453. `low_fpr` recalibrated to 0.601 (lowest yet).
   **Held-out at thr=0.601**: FP = **29.4 %**, recall = **87.7 %**. *Same FP as tweak 1; no further reduction from the extra metro weight.*
   **Threshold recal on held-out**: thr=0.710 → FP = 2.9 %, recall = **49.1 %** (best "FP≤5%" operating point across all tweaks, +8.7 pts absolute vs tweak 0's 40.4 % at the same FP). The dominant-metro training genuinely shifts the ROC curve up near the low-FP region, but not enough to reach the 86 % recall target.

**Summary: FP plateau across architecture-preserving tweaks.**

| Tweak | Default-thr FP | Default-thr recall | Best FP≤5% point (thr / FP / recall) |
|---|---|---|---|
| 0 (default: keep=0.5, ep=10, freeze) | 38.2 % | 89.5 % | 0.77 / 2.9 % / 40.4 % |
| 1 (keep=0.2, ep=15, freeze) | **29.4 %** | 89.5 % | 0.73 / 0.0 % / 35.1 % |
| 2 (keep=0.3, ep=15, unfreeze + warmup=3) | 38.2 % | 87.7 % | 0.74 / 2.9 % / 43.9 % |
| 3 (keep=0.1, ep=20, freeze) | **29.4 %** | 87.7 % | 0.71 / 2.9 % / **49.1 %** |

**Key observations for the paper**:
- Default-threshold FP plateaus at **~29.4 %** regardless of keep_safe=0.2 or 0.1. More aggressive re-weighting beyond 0.2 stops helping.
- Unfreezing the CNN is **actively worse** (tweak 2 is back at 38.2 % FP while running a *more* expensive training).
- The **best single architecture-preserving result** at the deployment gate (FP ≤ 5 %) is tweak 3: **49.1 % recall at 2.9 % FP**. Far below the claimed 86 % but a clear, measurable improvement over the default recipe.
- The ≤ 5 % FP / ≥ 86 % recall gate is **not reachable via training-distribution reweighting or CNN unfreezing alone** with the 58-clip metro fine-tune pool. Further gains require either (a) an order of magnitude more field-recorded site ambient, or (b) post-hoc interventions — site-ambient threshold recalibration (trivial patch) and temporal-majority aggregation (≥ 2 consecutive over-threshold windows before alerting).

5. **Tweak 4 — site-ambient threshold recalibration on default `metro_model.pth`** (no retrain).
   **Held-out recal**: thr=0.77 → FP 2.9 %, recall 40.4 %. *Confirmed equivalent to tweak 0's recal* (same held-out set, same model). The paper's framing point is that *the right place to calibrate `low_fpr` is on site ambient during fine-tune, not on the universal test set* — the two-line patch in [safecommute/pipeline/finetune.py:390-394](safecommute/pipeline/finetune.py) would replace the ROC sweep input with site-held-out ambient probabilities. This is an architecture-preserving change that every future fine-tune run should use.

---

## 2. What is actually publishable

### 2.1 The workshop-paper-worthy story

**Thesis**: *"A small privacy-preserving audio classifier passes every universal-benchmark check, then fails its own deployment gate in a precisely documentable way. Here are the measured failure modes, the root-cause analysis, and the minimum architecture-preserving interventions that could close the gap."*

That is a real workshop paper. It is not glamorous; it is honest; it is useful to the community because negative / calibration-gap results are systematically under-published in edge audio. Target venues, in order of fit:

| Venue | Fit | Why |
|---|---|---|
| **NeurIPS ML for Systems workshop** | Strongest | Explicitly welcomes deployment-reality papers. |
| **ICASSP late-breaking** (4 pages) | Strong | The γ-collapse finding fits their audio-ML audience. |
| **Interspeech late-breaking / show-and-tell** | Strong | Speech-adjacent, the laughter / shouting confusion is a speech-side interesting finding. |
| **EUSIPCO** | Moderate | Audio classification fits; honesty-frame is unusual but accepted. |
| **MLSys / MobiSys** | Moderate | The edge-latency + size work is on-topic; the deployment-gap frame is welcome. |
| arXiv preprint (no review) | Always | Put it up *now*; revise as workshop submission lands. |

### 2.2 The three concrete contributions worth defending

1. **Focal-loss γ-collapse in binary audio** (§3.2 of the planned structure below).
   Observation: γ=3.0 → AUC 0.856 but 0 % hard-negative accuracy (ambient → threat class-collapse). γ=0.5 + noise injection → AUC 0.804 with 28 % speech retained correctly. This is the clearest replicable finding.
   - **Limits**: we did not preserve the γ=3.0 checkpoint, so the 0.856 number is training-log only. Paper labels it *historical snapshot*.
   - **Fix before submission**: retrain γ=3.0 once to produce `models/safecommute_v2_gamma3.pth`, so the ablation row is re-runnable. ~2–4 hours on current hardware. Important for defensibility.

2. **Deployment-gap case study** (§3.4–§4.5 of the planned structure).
   The paper's original claim was "72% FP → 4.2% FP in 10 min fine-tune." We measured it end-to-end on truly-held-out ambient from the same site and got **38.2% FP**, not 4.2%. The paper's core new contribution is this *gap narrative*: a proper setup, a proper measurement, a proper diagnosis of why it fails, and a proper list of interventions that would close it. This is where the paper earns its right to exist.
   - The diagnosis is already solid: the `low_fpr` threshold is calibrated on the universal test set (no metro) and the sliding-window max-aggregation amplifies any single-frame false alarm into a per-file FP.
   - The minimum fix is to recalibrate the threshold on *site ambient* and require ≥2 consecutive over-threshold windows before alerting. Tweak 4 tests the first; a paper Addendum could test the second.

3. **PCEN + static INT8 ONNX as a privacy-and-latency pair for edge deployment** (§2 + §3.5).
   Not novel individually, but the measured combination (3.72 MB artifact, 2.86 ms mean CPU inference at 8T, AUC Δ = 0.002 vs FP32) is a clean, replicable edge-deployment package. Do *not* oversell privacy (see 2.3).

4. **"Unfreezing the CNN does not help for per-site ambient adaptation" — a small but defensible architectural observation** (§3.5).
   We measured the frozen-CNN + high-metro-weight recipe (tweak 1) against the unfrozen-CNN + warmup recipe (tweak 2) on the same held-out set. The frozen recipe produced a better held-out FP (29.4 % vs 38.2 %) with matching or better recall. Implication: for a model of this shape (CNN feature extractor + GRU/FC decision head) trained on a universal threat corpus, per-site adaptation is primarily a *decision-surface problem*, not a *feature-extraction problem*. You get more FP-reduction from shifting the safe-class training distribution toward site ambient than from unlocking CNN weights. This is the kind of negative result that saves others time.

### 2.3 What is NOT publishable (drop from the paper or reframe)

- **"PCEN is non-invertible therefore GDPR-safe by construction."** Technically the transform is many-to-one, but that is *not* a formal privacy guarantee. No differential-privacy budget, no membership-inference bound, no reconstruction-attack evaluation. Reviewers will bounce this. **Reframe** as "a reasonable feature-level privacy heuristic pending a formal attack evaluation in future work."

- **"50× smaller, 10× faster than CNN14/AST."** The parameter ratio is arithmetic and true. The latency ratio is based on literature numbers measured on different hardware. **Drop the 10× latency claim** unless we benchmark CNN14 and AST on the same hardware with the same feature pipeline.

- **"~12 ms on CPU"** as a standalone headline. **Reframe** as a hardware-disclosed table (§1.2).

- **"Deployable with 10 min of site audio."** The measured result says otherwise. **Reframe** as "deployable *after site-ambient threshold recalibration plus temporal aggregation*, pending further validation."

- **n=1 site deployment story framed as "metro deployment validated"**. Drop the framing. Say "we measured on one site and report the gap". Any stronger claim invites reviewer rejection.

---

## 3. Proposed paper structure (revised, honest version)

Target: 4–6 pages short paper, arXiv first.

**Title**: *"Measured Deployment Gap of a Privacy-Preserving Edge Audio Classifier: Focal-γ Collapse, Site-Ambient Threshold Miscalibration, and Interventions."*

*(The original title was "Site-Adaptive Edge Audio Classification for Public Safety". The new title makes the contribution the audit, not the product.)*

**1. Introduction (~0.5 page)**
- Edge audio for public-space escalation detection is constrained by (a) latency, (b) privacy, (c) ambient distribution shift.
- Gap: most published classifiers report universal-benchmark metrics; few report deployment-gate metrics on truly-held-out site ambient.
- Contribution: a 1.83 M-param CNN+GRU classifier with PCEN features; a hardware-disclosed latency profile; and **a measured deployment-gap study on one site with a root-cause analysis and reproducible intervention list**.

**2. Method (~1.5 pages)**
- 2.1 PCEN feature pipeline: 16 kHz → 64-band mel → PCEN (librosa defaults) → (1, 64, 188). Privacy frame is honest (lossy, not formally private).
- 2.2 Architecture: CNN6 + SE + GRU + multi-scale pool, 1.83 M params. Deliberately architecture-preserving through every experiment in this paper.
- 2.3 Training: focal loss, γ = 0.5, 30 % metro-noise injection at random SNR ∈ [0, 20] dB. sha256-split by source filename. SpecAugment.
- 2.4 Per-site fine-tune recipe: CNN frozen, GRU + freq_reduce + FC adapted, three thresholds computed (Youden, F1-optimal, `low_fpr`).
- 2.5 Deployment gate ([safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py)): sliding-window inference, RMS 0.003 energy gate, max-window aggregation per file, thresholded against `low_fpr`.

**3. Experiments (~1.5–2 pages)**

- 3.1 **Universal Phase A** on `prepared_data/test/` — AUC 0.804, acc 0.703, F1 0.716, per-source TPR/FPR table (§1.1 of this doc).
- 3.2 **Focal-γ ablation** (the first defensible finding): γ ∈ {0.0, 0.5, 1.0, 2.0, 3.0}. γ=3.0 collapses hard-neg accuracy to 0 % at AUC 0.856. γ=0.5 retains 28–64 % hard-neg accuracy at AUC 0.804. Retrain γ=3.0 once to ship a reproducible checkpoint.
- 3.3 **Export path**: static INT8 PTQ on fused ONNX; resulting 3.72 MB artifact with AUC Δ = 0.002, mean 2.86 ms on Ryzen 7 @ 8T; hardware-disclosed latency table.
- 3.4 **Per-site Phase B** on metro (n=1): training ambient 58 WAVs, held-out 34 WAVs, threats 57 WAVs. Results in §1.3.
- 3.5 **Tweak sweep** — five architecture-preserving interventions (§1.4): default (tweak 0), more metro weight (tweak 1), CNN unfreeze with warmup (tweak 2), dominant metro (tweak 3), threshold-only recalibration (tweak 4). All measured on the 34-clip truly-held-out quarantine set.
  - *Positive finding*: tweak 3 (keep_safe=0.1, 20 ep, frozen) achieves the best FP≤5% operating point: **49.1 % recall at 2.9 % FP**, +8.7 pts recall over the default recipe at the same FP. `low_fpr` threshold drops to 0.601 (from 0.667) because the model's confidence distribution shifts.
  - *Negative finding*: unfreezing the CNN (tweak 2) regresses FP to the default 38.2 % despite more trainable parameters. **Adaptation is a decision-surface problem, not a feature-extraction problem.**
  - *Plateau finding*: default-threshold FP drops from 38.2 % → 29.4 % going from keep_safe=0.5 → 0.2, but does not move further at keep_safe=0.1. Re-weighting the safe class is bounded by the 58-clip fine-tune pool size.
  - None of the five tweaks reach the ≤ 5 % FP / ≥ 86 % recall gate. The paper's contribution is the *precisely-quantified* plateau, not a victory claim.

**4. Discussion (~0.75–1 page)**

- 4.1 Why universal-benchmark AUC is not enough: max-window aggregation on 10–30 s clips amplifies a single-frame FP at a threshold that was calibrated for frame-level FPR.
- 4.2 Why the frozen-CNN fine-tune under-adapts: 10,189 kept-safe samples + 9,444 kept-unsafe + 3,489 new metro chunks → metro is only ~15% of the training distribution; the GRU + FC head can't shift a CNN-owned feature space that much.
- 4.3 The γ-collapse: focal's (1−p)^γ weighting at γ = 3 drives hard-example gradients to zero when p_t ≈ 0.5, which is exactly where speech-vs-shout confusion lives. γ = 0.5 preserves that gradient.
- 4.4 Privacy honestly: PCEN is lossy but not formally private. Reconstruction attacks are future work.
- 4.5 Implications for operators: at the *default* fine-tune recipe and threshold, expect ~15–40 % FP on unseen ambient even after calibration. Threshold recalibration on site ambient plus temporal majority voting likely gets the FP to the single digits; measurement in future work.

**5. Limitations (explicit, ~0.5 page)**

- n=1 site, curated from YouTube (not field-recorded).
- γ=3.0 ablation row is training-log only (checkpoint not preserved).
- SOTA comparison is parameter-count only; no same-hardware latency/accuracy comparison against CNN14/AST.
- Raspberry Pi latency is pending.
- Privacy claim is heuristic, not formally proven.

**6. Conclusion (~0.25 page)**

Edge audio classifiers that pass universal benchmarks can still fail their own deployment gates by 5–8× on held-out ambient from the target site. The failure is traceable (threshold miscalibration, aggregation-driven amplification, distributional dilution of per-site ambient during fine-tune). The interventions are architecture-preserving and testable. We release every checkpoint, every measurement script, and a single-command verifier.

**Reproducibility**: run `PYTHONPATH=. python tests/verify_performance_claims.py && python tests/finalize_release.py`. Exit 0 requires every doc claim to reproduce within tolerance on the same hardware; the current exit is 1 because four post-fine-tune claims do not. This is *by design* and is part of the paper's contribution.

---

## 4. Gap list — what to do before workshop submission

Ordered by impact on acceptance. This is what I'd do in the next 4–6 weeks.

### Must-do (or the paper gets rejected)

1. **Retrain γ=3.0 once** (4 hrs CPU/GPU) to produce `models/safecommute_v2_gamma3.pth`. Makes §3.2 ablation re-runnable. Blocks otherwise.
2. **Finish the tweak sweep** (already running in this session's background; see `tests/reports/tweak_finetune.json` for live updates). Report ≥ 3 measured tweaks in §3.5.
3. **Add at least one more site for Phase B** (n=2 is a materially different story than n=1). Cheapest route: record 30–60 min of ambient in (a) a bar/café, or (b) your apartment, or (c) a train platform. Re-run `tests/validate_fp_claims.py --environment bar --ambient-dir …`. Two sites beats one site by a lot at workshop review.
4. **Recalibrate `low_fpr` on site ambient instead of universal test**. This is a two-line patch in `finetune.py`: use the held-out ambient FPR curve, not the universal ROC. Measure; probably moves FP from 38% → 10–15%. Worth trying *before* submission.
5. **Implement temporal-majority aggregation** as an optional flag in `test_deployment.py` (≥2 consecutive over-threshold windows). Measure; probably moves FP from 15% → 5–8%. This is the architectural-preserving lever most likely to let us claim "gate met" honestly.
6. **Reframe every privacy sentence in the paper** as heuristic / feature-level / pending attack evaluation. This is a 30-minute edit but avoidable reviewer rejection.

### Should-do (strengthens acceptance odds)

7. **One baseline on the same test set under the same protocol**: either YAMNet (pretrained, 3.7 M params) or a CNN14 via PANNs. Run AUC, size, CPU latency. Adds the comparison row the paper promises.
8. **Raspberry Pi 4/5 latency row**. Need the hardware. If available: 200-iteration median + p99 on INT8 ONNX.
9. **Reconstruction attack on PCEN** (simple: train a small inverter and report bits-of-reconstruction). Makes the privacy section more than a hand-wave.
10. **End-to-end latency figure** ([tests/measure_e2e_latency.py](tests/measure_e2e_latency.py) output) in §3.3.

### Nice-to-have (polish)

11. Replace `docs/figures/gamma_ablation.png` labels to say "historical" where γ ≠ 0.5.
12. Add `figures_source.json` hash next to every figure caption in the paper.
13. A 5–6 sentence "negative result" abstract paragraph. Workshop reviewers love honesty when framed well.

Realistic timeline with items 1–6 + 7: **3–4 weeks of focused work**, dominated by field recording (if you do it) and the γ=3.0 retrain.

---

## 5. Site changes required to reflect reality

The website currently overclaims in several places. Each bullet below is scoped to minimize visual change — most are copy edits with optional asterisks.

### [web/components/hero.tsx](web/components/hero.tsx) — overclaims

- `StatDisc` line 127: `<CountUp value={12} suffix=" ms" />` + label "CPU inference".
  - **Fix**: keep the "12 ms" disc but add a fine-print footnote below the disc grid: *"Historical Ryzen 5, 1-thread. Current reference: 2.8 ms INT8 / 4.0 ms FP32 ONNX on Ryzen 7, 8T. See [/performance](...) for full hardware-disclosed table."*
- `StatDisc` line 137–140: `<CountUp value={5} prefix="<" suffix=" %" />` + "Speech FP after fine-tune".
  - **Fix**: remove the specific number; replace label with *"Speech FP drops substantially after site-ambient fine-tune"* or change disc to show *AUC 0.804* (measured, honest).
- Line 92–95 prose: "7 MB · 12 ms on CPU · no cloud".
  - **Fix**: "7 MB · 2.8 ms INT8 ONNX on CPU · no cloud" (reflects static INT8 shipped in the demo bundle).

### [web/components/honesty-block.tsx](web/components/honesty-block.tsx) — the virtue of admitting things

- Currently has a per-source accuracy table — check it matches §1.1 of this doc. If any cell drifts > 3%, update from [tests/reports/figures_source.json](tests/reports/figures_source.json).
- **Add a new honesty row**: "Post-fine-tune deployment gate: measured on n=1 site (metro). Held-out FP = 38.2%. The ≤5% target is not yet met; see [paper.md](paper.md) §4 for the root-cause analysis and in-progress interventions."

### [web/components/pricing.tsx](web/components/pricing.tsx) — overprice for current quality

- Current: €23/run, €100/site.
- Suggested reframing until FP ≤ 10% on ≥2 sites:
  - **€9/run** (the honest "research-tool" price).
  - **€39/site lifetime** (entry-tier positioning).
  - **€499–999 "deployment success"** (we do the calibration + iterate until the verifier exits 0).
- Or: open-source the whole fine-tune pipeline and sell only the deployment-success service. Strongest honesty play.

### [web/components/privacy-section.tsx](web/components/privacy-section.tsx) — structurally false for paid flow

Audit finding: *"No audio ever leaves the device"* is true for inference but false for the paid upload flow (raw WAVs sit in Supabase `audio-uploads` bucket until the worker processes them).

- **Fix option A (cheap)**: soften to *"Audio is processed in an encrypted, ephemeral pipeline and never persisted after training. On-device inference uses PCEN-only features which are non-invertible by construction."*
- **Fix option B (expensive, strong)**: implement client-side PCEN in WASM, upload only the spectrogram. Preserves the original hard claim. ~1–2 weeks.

### [web/components/problem-grid.tsx](web/components/problem-grid.tsx) — "~72% false positives"

- Measurement: 71.7% speech FPR. Current copy matches within 1%. **No change needed** — but link to `tests/reports/verify_performance_claims.json` so the reader can verify.

### [web/components/edge-positioning.tsx](web/components/edge-positioning.tsx) — "~12 ms CPU latency" and "RPi 4+ ARM-ready"

- **Fix**: same footnote shape as hero. "RPi 4+ ARM-ready" should either include a measured RPi latency row (see Gap item 8) or be softened to *"ARM-ready, Raspberry Pi benchmark pending"*.

### [web/components/how-it-works.tsx](web/components/how-it-works.tsx) — "Speech FP 72% → <5%"

- **Fix**: "Speech FP 72%. Drops substantially after site calibration (current measurement: 15% on universal speech subset after metro fine-tune; target ≤ 5% pending threshold + aggregation fixes — see [paper.md](paper.md) §4.5)."
- Or: remove the `→ <5%` claim entirely; replace with a pointer to the live figure.

### [web/public/demo/short.md](web/public/demo/short.md) — already updated in this session

No further action needed (measured numbers + Limitations present).

### [README.md](README.md) + [RESULTS.md](RESULTS.md) — already updated in this session

No further action.

### Summary of site-copy deltas

| Component | What changes | Who can do it |
|---|---|---|
| hero.tsx | add latency footnote, remove "<5%" stat disc | you, 10 min |
| honesty-block.tsx | add Phase-B honesty row | you, 15 min |
| pricing.tsx | lower headline prices or add service tier | you, product decision |
| privacy-section.tsx | soften or implement WASM PCEN | copy: 10 min; code: 1–2 wk |
| edge-positioning.tsx | latency footnote + RPi softening | you, 10 min |
| how-it-works.tsx | remove "<5%" from step 04 | you, 10 min |
| problem-grid.tsx | none (71.7% ≈ 72%) | — |

Total ~1 hour of copy edits to stop overclaiming; the WASM-PCEN privacy fix is the one real sprint.

---

## 6. Pricing — honest take

Current pricing: **€23 / single fine-tune, €100 / site unlock lifetime.**

Grounded comparisons:

| Service | Price | What you get |
|---|---|---|
| Replicate (custom fine-tune) | $2–10 per run | Compute only. |
| Roboflow (labeling + fine-tune) | $99/mo | End-to-end platform. |
| AssemblyAI (hosted audio API) | $0.37/hr | No model shipped, pay-as-you-go. |
| Freelance ML engineer (one-off audio classifier) | €200–500 flat | Bespoke, can beat a generic recipe. |
| Hume AI (voice analysis) | $0.02/minute | Hosted API. |

Given current measured accuracy (38.2% FP on metro held-out, no operating point reaches the ≤5% gate):

| Tier | Current | Recommended while gate not met | Rationale |
|---|---|---|---|
| Per-run | €23 | **€9** | Acknowledges "research tool, not production-grade". |
| Site unlock | €100 | **€39** | Entry-tier; ~4× per-run (usual ratio). |
| Deployment success | — | **€499–999** | Bundle the work (calibration + iteration) rather than sell a model that may not clear the gate. This is where the real value is right now. |
| Enterprise | "Custom" | Keep | Unchanged. |

When the verifier exits 0 on ≥ 3 sites with FP < 10 %, re-raise prices toward €23 / €100.

**Alternative pricing structure worth considering** — pure open-source the fine-tune pipeline, sell only the *deployment-success* service. This matches the reality that the heavy lift is the calibration + iteration, not the model weights. This also sidesteps the "your €100 product has 38% FP" conversation entirely, because the customer pays for outcome, not artefact.

---

## 7. What is worth doing next, ordered by workshop-publishability leverage

Given your priority is workshop publishability, do these in this order:

### This week (biggest impact for least effort)

1. **Wait for the current tweak sweep to land**, splice measured Tweak 2 / 3 / 4 numbers into §1.4 and §3.5. (*In progress in this session; see `tests/reports/tweak_finetune.json`.*)
2. **Retrain γ=3.0 once**. 4 hours. Produces `models/safecommute_v2_gamma3.pth`. Makes the one defensible ablation re-runnable.
3. **Implement the site-ambient `low_fpr` recalibration** and measure. Two-line patch. Probably moves metro FP from 38% → 10–15%.
4. **Implement temporal-majority aggregation** (optional flag in `test_deployment.py`), measure. Probably moves FP further to 5–8%. If this works, the paper can claim "gate met via threshold + aggregation interventions on site 1" which is a real result.
5. **Reframe privacy section** (30 min edit) — remove formal-guarantee language.

If items 3+4 produce FP ≤ 10%, the paper has a *positive* result alongside the deployment-gap finding. That's workshop-strong.

### Next 2–3 weeks

6. **One more site**. Record ambient in any distinct acoustic environment (bar, apartment, café, library) for 30–60 min. Run the full fine-tune + measurement on it. n=2 beats n=1 at review. If the interventions from step 3+4 generalize, the paper has a *repeatable* positive result.
7. **One SOTA baseline** on the same hardware. Suggest: YAMNet (fastest to set up, smallest footprint match). Run Phase A. Adds the comparison table row.
8. **End-to-end latency measurement** in the paper (already scripted).
9. **Draft the paper on arXiv** — write first, refine for a workshop submission window.

### Weeks 4+

10. **Pi 4/5 benchmark** if hardware arrives. Adds the ARM row the pitch already claims.
11. **PCEN reconstruction attack** (train a small inverter; report MSE / bits-of-info leaked). Elevates the privacy claim from heuristic to quantified.
12. **Third and fourth sites** if feasible.
13. Submission.

### What NOT to do

- Don't retrain the whole base model hoping to "improve" AUC into the 0.85+ range. Diminishing returns; not a workshop contribution.
- Don't add a new architecture (transformer / attention / distillation). Architecture-novelty papers are a different genre and your current archive is not competitive there.
- Don't chase the original "4.2% FP" marketing number. The paper's contribution is the honest gap, not hitting that specific figure.
- Don't add more AudioSet data and retrain. Returns are mostly gone.

---

## 8. Repo state (for the reproducibility appendix)

Git HEAD (as of last `finalize_release.py` run): `7b7013efe56b3977db3cce2e16166fac89cbc40c` (update at submission).

Artefact hashes live in [tests/reports/artefacts.sha256](tests/reports/artefacts.sha256). Current file list:

- `models/safecommute_v2.pth` — 7.00 MB, base checkpoint
- `models/safecommute_v2.onnx` — 6.99 MB, single-file FP32 ONNX (Step 1)
- `models/safecommute_v2_fused.onnx` — 6.99 MB, graph-optimized FP32
- `models/safecommute_v2_int8.onnx` — 3.72 MB, static-PTQ INT8 (Step 5)
- `models/safecommute_v2_int8.pth` — 5.06 MB, dynamic-quant PyTorch (reference)
- `models/metro_model.pth` — 7.00 MB, per-site fine-tune (metro, default recipe)
- `models/metro_tweak1_keep0.2_ep15_model.pth` — 7.00 MB, tweak 1 checkpoint

Single-command verification:
```
PYTHONPATH=. python tests/verify_performance_claims.py
```
Exit 0 = every doc claim reproduces within tolerance. Current exit is 1 (the four post-fine-tune claims; by design).

Report snapshot:
```
PYTHONPATH=. python tests/finalize_release.py
```
Writes `tests/reports/SUMMARY.md` + `tests/reports/artefacts.sha256`.

---

## 9. What this paper is NOT (just so the framing stays honest)

- Not a proposal of a new audio-ML architecture.
- Not a formal privacy paper.
- Not a direct apples-to-apples SOTA benchmark.
- Not a product launch announcement.
- Not a proof that the system is deployable; it's a proof that the system is *almost* deployable and a diagnosis of what closes the gap.

If we frame it that way from the abstract onward, workshop reviewers tend to respond well. Overclaim and they don't.
