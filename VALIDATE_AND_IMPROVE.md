# Validate & Improve

> **Status update 2026-04-27 (production live)** — the upload → pay → fine-tune → download flow is live on <https://safecommute-ai.vercel.app> in Stripe test mode. Migration 0002 applied, worker running self-hosted as a systemd unit on the Ryzen 7 7435HS dev box, Vercel production carries the new download route + 3 dashboard buttons + reframed privacy section. End-to-end smoke test passed on iPhone Safari 2026-04-27. Operational state owned by [DEPLOYMENT_NEXT_STEPS.md](DEPLOYMENT_NEXT_STEPS.md). Operational caveat: GitHub → Vercel auto-deploy is broken (three pushes between 2026-04-22 and 2026-04-27 left production frozen), so until the integration is reconnected every redeploy is `cd web && vercel --prod` by hand. What's left for "first real paying customer": Stripe live-mode flip (DEPLOYMENT_NEXT_STEPS.md §5). What's left for workshop submission: ≥ 2 more recorded sites (DEPLOYMENT_NEXT_STEPS.md §7). See §15 below for the closing landing notes.
>
> **Status update 2026-04-22 (Phase 1 + 2 + 4 complete)** — §14 at the bottom documents what landed this session. The gap list from 2026-04-21 is now a ✅-mostly document: the two architecture-preserving levers (site-ambient threshold recalibration + temporal-majority aggregation) are implemented and close the FP ≤ 5 % gate on n=1 site (FP 0.0 % / recall 78.9 % on the 19-wav metro eval half — the remaining shortfall is 9 pts of recall, not 30+ pts of FP). γ=3.0 retrain done (AUC 0.791 measured — **material correction** from the 0.856 training-log snapshot; per-source hard-neg breakdown confirms the γ-collapse pattern). PANNs CNN14 SOTA baseline measured on same hardware (44.7× smaller / 22.7× faster @ 8T / 13.9× faster @ 1T). Fine-tune worker + download route + privacy fix + .env.example + systemd unit all landed — paying customer can now complete an upload → pay → download flow (product side was explicitly out of scope on 2026-04-21). Only remaining blocker for a defensible workshop submission: ≥ 2 more sites of field-recorded ambient (user-side only — infra is in place).
>
> **Status update 2026-04-21 (initial session end)** — the plan below was executed end-to-end. Sections that got fully done are marked inline; the consolidated status for the 2026-04-21 session lives in §12. The authoritative on-disk numbers now live in [`tests/reports/verify_performance_claims.json`](tests/reports/verify_performance_claims.json) (Phase A + latency), [`tests/reports/phase_b_metro.json`](tests/reports/phase_b_metro.json) (Phase B on n=1 site), and [`tests/reports/tweak_finetune.json`](tests/reports/tweak_finetune.json) (now a 6-attempt tweak sweep — see §14 for tweak 5). When those files disagree with a prose claim in this document, the JSONs win. Full narrative: [`paper.md`](paper.md). Release summary: [`tests/reports/SUMMARY.md`](tests/reports/SUMMARY.md).

Living checklist for hardening SafeCommute AI toward two objectives:

1. **An applied ML paper** — every quantitative claim in the paper must come from a re-runnable script in this repo. No hand-figured numbers.
2. **A public fine-tuning website** — users upload ambient audio from their environment, get a personalized model back. That model must hit the stated <5% FP / ≥86% recall claims on real data, on the hardware they deploy to (typically a small Linux box or a Raspberry Pi).

Everything below is scoped to serve those two endpoints.

---

## 0. TL;DR Verdict (Full-Repo Audit, 2026-04-21)

**One line:** A real ML idea with one publishable finding (focal-loss γ collapse), wrapped in marketing claims that the codebase does not yet substantiate, and a paid product whose execution backend doesn't exist.

**What is genuinely solid (reproducible from on-disk artefacts):**
- Architecture (1.83M params, 7 MB float32, PCEN input contract).
- Base discriminative metrics on `prepared_data/test/` (AUC 0.804, acc 70.3%, F1 0.716, per-source TPR/FPR).
- The γ=0.5 vs γ=3.0 finding qualitatively (γ=3.0 zeroes hard-negatives in binary audio classification).
- PCEN as a structural privacy mechanism for the inference path (not the upload path — see below).
- Source-level SHA256 splitting (no random per-sample leakage).

**Critical gaps blocking paper submission:**
1. **Post-fine-tune claims (5% FP, 4.2% speech FP, 86% recall) have never been run end-to-end.** No `<site>_model.pth` exists on disk. `tests/validate_fp_claims.py` Phase C measures the wrong dataset (universal test set instead of held-out site ambient). See §5.0.
2. **The 12 ms latency claim does not reproduce** on the only hardware where it has been measured (Ryzen 7 7435HS: 108 ms FP32 / 65 ms ONNX at default threads). Hardware/threads/BLAS never disclosed in any doc. See §5.6, §5.5.
3. **The 41% leakage figure is a hand-typed literal** at [`scripts/generate_pitch_figures.py:267-268`](scripts/generate_pitch_figures.py#L267-L268). Never derived from the model. Actual measurement: 34.2% FPR. See §5.7.
4. **`models/safecommute_v2_gamma3.pth` does not exist**, so the AUC=0.856 number in the γ ablation is not currently re-runnable.

**Critical gaps blocking paying customers:**
1. **No fine-tune worker.** Jobs queue into Supabase `finetune_jobs` with `status='queued'` but nothing consumes the queue. A customer can pay €23 or €100 today and get a permanently queued job. See §10.1.
2. **No model download route.** `/api/finetune/[id]/download` does not exist; even if a job ran, customers couldn't retrieve the artefact. See §10.2.
3. **The web "no audio leaves the device" copy is structurally false** for the paid flow — clips upload to Supabase Storage *before* PCEN is applied server-side. The privacy differentiator is being mis-sold. See §10.3.
4. **No SDK or integration guide.** Customers who download the demo `.pth` have no documented path from "I have the file" to "it runs in my product".

**Realistic timelines:**
- 2–3 weeks to an honest arXiv preprint (limitations explicit, latency hardware disclosed).
- 6–8 weeks to a defensible workshop submission (paper items 1–7 below, plus §6).
- 4–6 weeks to a launchable v1 paid product (sellability items 1–4 below).

The paper-side fix list is already in §5 + §6 + §7. The product-side fix list lives in the new §10 and §11 below.

---

## 1. How the claims actually map to the pipeline

This is critical and easy to misread. The repo makes **three different kinds** of performance claim, evaluated in **three different settings**:

**(A) Universal / base-model claims** — measured on `prepared_data/test/` (the held-out AudioSet + YouTube + ESC-50 + violence dataset mix). These evaluate the model's general threat/non-threat discrimination. The base model is **NOT expected to be deployable from these numbers alone**; they describe raw discriminative ability.
- AUC 0.804, accuracy 70.3%, F1 0.716
- Per-source TPR (yell 90.6%, scream 79%, shout 65%, etc.)
- Per-source FPR — speech 72%, laughter 82%, crowd 58%, metro 35%
- **The 72% speech-FP is a universal claim on the base model**: "out of the box, speech confuses it." That's the motivation for the fine-tuning pipeline, not a failure of the model.

**(B) Deployment claims** — measured by [`safecommute/pipeline/test_deployment.py`](safecommute/pipeline/test_deployment.py) on *site-specific* `.wav` files via sliding-window inference with the fine-tuned model's `low_fpr` threshold. These are what a deployed system actually experiences.
- Threat detection ≥ 90% ([test_deployment.py:11](safecommute/pipeline/test_deployment.py#L11))
- FP rate ≤ 5% ([test_deployment.py:14](safecommute/pipeline/test_deployment.py#L14)) — **measured on the target site's held-out ambient `.wav`s**
- Latency mean < 15 ms, p99 < 30 ms
- Model size ≤ 10 MB FP32, ≤ 6 MB INT8
- Bit-identical outputs across repeated inference
- Silence energy-gated as safe

**(C) Narrative / marketing claims** — specific before/after numbers cited in the paper plan, pitch deck, and homepage. These are the load-bearing numbers a reader will pull out and quote.
- Post-fine-tune **speech FP drops 71.7% → 4.2%** after 10 minutes of site ambient ([update.md:116](update.md#L116), [docs/gamma_prompt.md:85](docs/gamma_prompt.md#L85))
- Post-fine-tune **threat recall maintained at ~86%** ([paper.md:34](paper.md#L34))
- Base-model **threat recall ~82%** (micro, confusion-matrix reading) ([update.md:114](update.md#L114))
- Base-model **41% safe→unsafe leakage driven by speech** ([update.md:114](update.md#L114)) — *my measurement shows 34.2% overall FPR, so this one needs either validation or correction before the paper*
- Gamma ablation: **γ=3.0 → AUC 0.856 but 0% hard-negative accuracy; γ=0.5 + noise → AUC 0.804** ([paper.md:32](paper.md#L32), [update.md:102](update.md#L102))
- SOTA footprint comparison: **~50× smaller, ~10× faster than CNN14/AST** ([paper.md:35](paper.md#L35), [docs/gamma_prompt.md:82](docs/gamma_prompt.md#L82))
- CPU latency **~12 ms** on Ryzen 5, 1 thread ([short.md:44](web/public/demo/short.md#L44), [README.md:20](README.md#L20), [RESULTS.md:12](RESULTS.md#L12))

**The 5% FP claim is a (B)-class deployment claim, not an (A)-class universal claim.** It is achievable only when:
1. The model has been fine-tuned on ≥30 min of that site's ambient (per the now-deleted `DEPLOY.md` — the requirement now lives only implicitly in [finetune.py](safecommute/pipeline/finetune.py); this doc should restate it publicly).
2. The fine-tuned `low_fpr` threshold is applied.
3. The FP rate is measured on held-out ambient from *that same site* — not on a universal benchmark.

Out of the box, the base model's FP rate is 35–82% depending on source. It is not intended to be deployed as-is.

## 2. Where each claim shows up (claim surface audit)

The same numbers appear across multiple files, which means a change to the model silently contradicts up to 9 other docs. The paper-submission and website-deploy gates need to check all of them in lockstep.

| Source file | What it asserts |
|---|---|
| [RESULTS.md](RESULTS.md) | AUC, accuracy, F1, per-source TPR (4 sources) and FPR (4 sources), speech-FP summary |
| [README.md](README.md) | Latency 12 ms, AUC 0.804, "deploy at 12ms on CPU" |
| ~~DEPLOY.md~~ | **Deleted** (was the canonical source for post-fine-tune FP ≤ 5%, threat detection ~86%, latency 12 ms, RPi 4+). These claims now have no doc home and must be restored — either re-create DEPLOY.md or merge into a single RESULTS.md §Deployment section. |
| [CLAUDE.md](CLAUDE.md) | 1.83M params, 7 MB float32, ~12 ms CPU |
| [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py) | Deployment gate: TPR ≥ 90%, FP ≤ 5%, latency ≤ 15 ms / p99 ≤ 30 ms, size ≤ 10 MB / ≤ 6 MB |
| [safecommute/export.py:23](safecommute/export.py#L23) | Export target: latency < 15 ms |
| **[scripts/generate_pitch_figures.py](scripts/generate_pitch_figures.py)** | **Hardcoded confusion matrix `[[0.59, 0.41], [0.18, 0.82]]` (line 267-268), subtitle "AUC 0.80 - acc 70% - that 41% slip is the speech problem" (line 303), per-source accuracy array literals (lines 147-154), SOTA comparison (lines 85-88: YAMNet 3.7M/50ms/0.306, CNN14 80M/150ms/0.431, AST 86M/200ms/0.485).** This is where the 41% comes from — it is a manual entry in a plot script, not a measurement. |
| [web/public/demo/short.md](web/public/demo/short.md) | "~12 ms on a single CPU core, Ryzen 5, 2024"; 72% base speech-FP; ~86% post-fine-tune recall |
| [web/components/edge-positioning.tsx](web/components/edge-positioning.tsx) | "~12 ms CPU latency"; "RPi 4+ ARM-ready" |
| [web/components/hero.tsx](web/components/hero.tsx) | "7 MB"; "12 ms"; "no cloud" |
| [web/components/how-it-works.tsx](web/components/how-it-works.tsx) | "1.83 M params, 7 MB float32, ~12 ms CPU" (step 03); "Speech FP 72% → <5%" (step 04) |
| [web/components/problem-grid.tsx](web/components/problem-grid.tsx) | "~72% false positives" |
| [update.md](update.md) | Speech FP 71.7% → 4.2%; base recall 82%; leakage 41%; ~50× smaller / ~10× faster than SOTA |
| [paper.md](paper.md) | Paper plan; gamma ablation (0.856 vs 0.804); SOTA comparison; post-fine-tune 86% recall |
| [docs/gamma_prompt.md](docs/gamma_prompt.md) | Pitch narrative; repeats the 71.7% → 4.2% and 50×/10× comparisons |
| [selling.md](selling.md) | Business claim: free-tier is 72% FP, paid-tier is <5% FP |

**Total: 15 claim-duplication surfaces.** Any change to a model metric must land in all of them, or `verify_performance_claims.py` (once written) will flag the inconsistency.

## 3. What we've validated so far

| Claim | Kind | Status | Evidence |
|---|---|---|---|
| Base AUC = 0.804 | A | **Validated** (0.804) | `tests/analyze_model.py` |
| Base accuracy = 70.3%, F1 = 0.716 | A | **Validated** | `tests/analyze_model.py` |
| Params 1.83M, size 7 MB | static | **Validated** | `tests/analyze_model.py` |
| Base per-source TPR (yell, scream, shout, yt_scream) | A | **Validated** | `tests/analyze_model.py` |
| Base per-source FPR (speech, crowd, laughter, metro) | A | **Validated** | `tests/analyze_model.py` |
| Base speech-FP ≈ 72% | A / C | **Validated** (71.7%) | `tests/validate_fp_claims.py --phase a` |
| Base recall = 82% (micro, confusion-matrix) | C | **Validated** (~80.9% sample-weighted) | `tests/analyze_model.py` |
| **Base leakage = 41% (safe→unsafe)** | **C** | **Source identified — it is a HARDCODED LITERAL in a plot script**. `cm = np.array([[0.59, 0.41], [0.18, 0.82]])` at [generate_pitch_figures.py:267-268](scripts/generate_pitch_figures.py#L267-L268); subtitle text references 41% at line 303. My measurement shows overall FPR = 34.2% at thr=0.5. The 41% in the plot was never derived from the model. Either the matrix needs to be regenerated from real inference, or the narrative needs to match the measured number. | [scripts/generate_pitch_figures.py:267](scripts/generate_pitch_figures.py#L267), `tests/analyze_model.py` |
| Post-fine-tune **speech FP 71.7% → 4.2%** | C | Never verified end-to-end | *pending* |
| Post-fine-tune **threat recall ≥ 86%** | B / C | Never verified end-to-end | *pending* |
| Post-fine-tune **overall FP ≤ 5%** on site ambient | B | Never verified end-to-end | *pending* |
| Gamma ablation (γ=3.0 → 0.856, γ=0.5 → 0.804) | C | Half validated (0.804 is current model); γ=3.0 requires retraining with old config | *not re-runnable without code* |
| SOTA footprint: ~50× smaller, ~10× faster than CNN14/AST | C | Footprint math checks (CNN14 ≈ 80M vs 1.83M ≈ 44× ✓); latency comparison has no measured CNN14/AST row in the repo | *pending* |
| Latency **~12 ms** on Ryzen 5, 1 core | C | **Does not reproduce** — 108 ms FP32, 65 ms ONNX on Ryzen 7 7435HS, default threads; 325 ms ONNX at 1 thread | `tests/measure_latency.py` |
| Deployment latency ≤ 15 ms / p99 ≤ 30 ms | B | **Fails on this machine** (108 ms mean) | `safecommute/export.py` |
| RPi 4+ ARM-ready | B | **Aspirational** — no Pi measurement, no ARM artifact shipped | — |

**What I got wrong before**: [tests/validate_fp_claims.py](tests/validate_fp_claims.py) Phase C measures the fine-tuned model's FPR on the **universal** base test set (`prepared_data/test/0_safe/`), which is the wrong denominator. The 5% / 4.2% claims are about the **site's own held-out ambient**, evaluated the way [test_deployment.py](safecommute/pipeline/test_deployment.py) evaluates it. This needs fixing (see 5.0 below).

## 4. What's broken or weak on the release path

- **Demo zip ships `safecommute_v2.pth`** ([safecommute-v2-demo.zip](web/public/demo/safecommute-v2-demo.zip)) — users need full PyTorch to run it.
- **ONNX export produces a 100 KB skeleton** with weights stored as external data (PyTorch 2.11 dynamo default). The file alone does not load on another machine.
- **INT8 dynamic quantization is a regression** — [safecommute/export.py:82-98](safecommute/export.py#L82-L98) only quantizes Linear + GRU. The 3 CNN blocks dominate compute; INT8 runs 20% *slower* than FP32 in the repo's own benchmark.
- **Per-source FPR table in [RESULTS.md:25-31](RESULTS.md#L25-L31) lists 4 of 16 safe sources.** Laughter (82% FPR) is the top false-alarm and is omitted. Post-fine-tune-focused story needs the full table.
- **The 41% leakage figure in [update.md:114](update.md#L114) doesn't match my measurements (34.2%).** Either a different threshold/subset was used, or the figure is stale. Needs clearing up before the paper.
- **No claim-verification script** — a single `tests/verify_performance_claims.py` needs to consolidate the pieces (see 6).
- **No run of [test_deployment.py](safecommute/pipeline/test_deployment.py) exists on disk** — no generated `<site>_model.pth`, so the 5%/4.2%/86% claims have never been independently re-verified. This is the biggest paper-readiness gap.
- **[scripts/generate_pitch_figures.py](scripts/generate_pitch_figures.py) does not read the model.** Every bar, every matrix cell, every subtitle number is a hand-written literal. The plots drift silently when the model changes. The script must be rewritten to consume `tests/analyze_model.py` output (or a cached JSON blob thereof) instead of hard-coding.
- **Repo hygiene gaps that affect reproducibility/publishability**:
  - `DEPLOY.md` is **staged for deletion** — the canonical source for the post-fine-tune claims. Its content (threat detection ~86%, FP ~7% / <5% with optimized threshold, RPi 4+ hardware requirement, 12ms CPU claim) is not carried anywhere else on the tracked-files path. Either restore the file or merge its content into RESULTS.md §Deployment.
  - `next_steps.md` was renamed to `paper.md` (the .gitignore still lists `next_steps.md`, but the file doesn't exist). `paper.md` is **untracked**.
  - `tests/` directory is **untracked** — none of `validate_fp_claims.py`, `analyze_model.py`, `measure_latency.py`, or `measure_latency_opt.py` is in git. A fresh clone doesn't have them.
  - `models/` directory is **untracked** (expected — large binaries should be released separately or via Git LFS).
  - `VALIDATE_AND_IMPROVE.md` (this file) is **untracked**. Decide whether it belongs in git or in `.gitignore` alongside CLAUDE.md — either is defensible, but the current ambiguity is bad.
  - [.gitignore:184-191](.gitignore#L184-L191) excludes `CLAUDE.md`, `selling.md`, `update.md`, `next_steps.md`, `docs/gamma_prompt.md` as "internal planning / strategy / pitch docs (not for public repo)". A fresh clone therefore **won't have the operations manual (CLAUDE.md) or the pricing/sales model (selling.md)**. This is fine if intentional — but the reviewer/collaborator onboarding flow needs to be explicit about how they get these files. Consider a private git submodule or an encrypted bundle for internal-only docs.
  - No `.env.example` for the web stack. Setting up `/web/` on a new machine requires walking through [DEPLOY_WEB.md §3-4](DEPLOY_WEB.md) manually — error-prone and violates reproducibility.

---

## 5. Changes to implement (ranked by impact on paper + demo)

Each item lists: what it does, why, how to implement, how to verify it worked.

### 5.0 Fix [tests/validate_fp_claims.py](tests/validate_fp_claims.py) Phase C to match the claim it's checking

**Why**: the script currently measures fine-tuned FPR on the universal test set, but the 5% / 4.2% FP claims are about held-out site ambient. As written, the script can pass while the actual deployment claim fails, or vice versa.

**How**:
- Take `--ambient-dir raw_data/<site>/` and `--threat-dir raw_data/<threats>/` as arguments.
- Split the `.wav` files 80/20 deterministically by filename SHA256 (ambient only — threat dir is held-out as-is).
- Run [finetune.py](safecommute/pipeline/finetune.py) on the 80% ambient.
- Delegate FP measurement to [test_deployment.py](safecommute/pipeline/test_deployment.py)'s `test_false_positive` against the held-out 20%, with the `low_fpr` threshold from the generated `thresholds.json`.
- Delegate threat detection to `test_threat_detection` against the full threat dir.
- Check both against the claims: FP ≤ 0.05 AND TPR ≥ 0.86 (matches [paper.md:34](paper.md#L34) paper target; tolerance = 3 samples).
- Also measure speech-FP specifically after fine-tune and check against the 4.2% claim (tolerance ≤ 8% — the figure comes from a specific figure generation run and the measurement has noise).

**Verify**: script output like:
```
Site: my_metro  (fine-tune: 45 wavs, held-out: 13 wavs)
  FP rate on held-out ambient:   0.038   [PASS ≤ 0.05]
  Threat recall on screams:      0.906   [PASS ≥ 0.86]
  Speech FP post-fine-tune:      0.061   [PASS ≤ 0.10]
```

### 5.1 Fix the ONNX export so it ships as a single file

**Why**: without this, every downstream optimization (INT8, ARM, browser) is blocked. Current `models/safecommute_v2.onnx` is unusable on any other machine.

**How**: in [safecommute/export.py](safecommute/export.py), either pass `dynamo=False` to `torch.onnx.export` (legacy exporter, inlines weights) or keep the dynamo path and post-process:
```python
import onnx
m = onnx.load(path, load_external_data=True)
onnx.save(m, path, save_as_external_data=False)
```
Remove `dynamic_axes={"mel_spectrogram": {0: "batch"}}` — batching is not needed for the edge use case.

**Verify**: `ls -lh models/safecommute_v2.onnx` shows ~7 MB. From a fresh venv:
```bash
python -c "import onnxruntime as ort; ort.InferenceSession('models/safecommute_v2.onnx')"
```
No error, no "missing external data" warning.

### 5.2 Static INT8 PTQ on Conv layers (the actual latency lever)

**Why**: Conv2d accounts for >80% of compute. Quantizing to INT8 uses VPDPBUSD on x86 and SDOT on ARM — real 2-4× wins. Linear+GRU dynamic quant does nothing useful and hurts.

**How**: new script `safecommute/export_quantized.py`:
```python
from onnxruntime.quantization import (
    quantize_static, QuantFormat, QuantType, CalibrationDataReader,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

class SpecReader(CalibrationDataReader):
    def __init__(self, n=128):
        # Load n PCEN tensors from prepared_data/test across both classes
        # so the calibration distribution matches deployment.
        ...
    def get_next(self):
        return next(self._iter, None)

quant_pre_process("models/safecommute_v2.onnx",
                  "models/safecommute_v2.prep.onnx")
quantize_static(
    "models/safecommute_v2.prep.onnx",
    "models/safecommute_v2_int8.onnx",
    calibration_data_reader=SpecReader(),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv", "MatMul"],   # skip GRU
    calibrate_method=CalibrationMethod.MinMax,
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
)
```
If the quantizer still touches the GRU, exclude node names explicitly via `nodes_to_exclude=[...]`.

**Verify**:
- File size ≤ 2.5 MB.
- `tests/validate_fp_claims.py` against the INT8 model shows speech FPR and AUC within 2 percentage points of the FP32 baseline.
- `tests/measure_latency.py` with the INT8 ONNX shows ≤ 40 ms median on this machine, ≤ 15 ms on the paper's target Ryzen 5.

### 5.3 Conv+BN folding at export time

**Why**: free ~10-15% speedup, smaller graph, deterministic. No calibration, no accuracy risk.

**How**: after `model.eval()` and before ONNX export:
```python
model = torch.ao.quantization.fuse_modules(
    model,
    [["block1.conv", "block1.bn"],
     ["block2.conv", "block2.bn"],
     ["block3.conv", "block3.bn"]],
    inplace=False,
)
```

**Verify**: `tests/analyze_model.py` AUC before/after is identical (fusing is arithmetically exact in FP32 inference).

### 5.4 Ship the right thing in the demo zip

**Why**: current zip is a 7 MB `.pth` that requires PyTorch. Change it to a zero-PyTorch-dependency bundle.

**How**: edit [web/scripts/build-demo-bundle.mjs](web/scripts/build-demo-bundle.mjs) to include:
- `safecommute_v2_int8.onnx` (from 5.2)
- `infer.py` — ~40 lines, loads the ONNX, reads a `.wav`, prints `safe`/`unsafe` + probability. Deps: `numpy`, `scipy`, `onnxruntime`. **No** PyTorch, **no** librosa.
- `PCEN.py` — minimal pure-scipy PCEN (~30 lines).
- `README.md` — the existing `short.md` with an added "Install" block: `pip install numpy scipy onnxruntime`.

**Verify**: on a fresh container with Python 3.11 and no PyTorch, `pip install ...` + `python infer.py sample.wav` runs end-to-end in under 5 seconds.

### 5.5 Benchmark on Raspberry Pi for real

**Why**: "RPi 4+ ARM-ready" is currently aspirational. The paper needs a measured row in the latency table.

**How**:
- Borrow or buy a Pi 4 and Pi 5. Copy the INT8 ONNX + `infer.py` (scp).
- `pip install onnxruntime numpy scipy`; `python bench.py`.
- Log median / p99 over 200 iterations at 1, 2, 4 threads.
- Record `/proc/cpuinfo`, governor, ambient temperature, heatsink status.

**Verify**: add the measured row to [RESULTS.md](RESULTS.md). If Pi 4 exceeds 1 s/inference even with INT8, drop the "RPi 4" claim or invest in ExecuTorch/XNNPACK.

### 5.6 Separate "inference latency" from "end-to-end latency"

**Why**: the 12 ms claim measures the model forward pass only. librosa PCEN takes ~80 ms on a Pi 4 — the actual end-to-end latency is dominated by preprocessing. Paper and docs should report both.

**How**: `tests/measure_latency.py` already measures model-only. Add `tests/measure_e2e.py` that times raw PCM → PCEN → normalize → model → softmax. Report both numbers.

### 5.7 Reconcile or fix the 41% leakage figure in [update.md:114](update.md#L114)

**Why**: my measurements (overall FPR = 34.2% at thr=0.5) don't match. If the 41% came from a different threshold or subset, that needs documenting. Otherwise the doc is stale.

**How**: either (a) identify the subset/threshold the figure was originally computed on and annotate the doc, or (b) replace it with the current measurement. Do before paper submission — a reviewer will run the numbers.

**Verify**: the `verify_performance_claims.py` script explicitly checks the leakage figure matches an identified, documented configuration.

---

## 6. New file to write: `tests/verify_performance_claims.py`

One script, one command, one pass/fail report covering **every numerical performance claim** across the 12 files in Section 2. Scope is performance only — not privacy, not infra, not website-flow.

The script has three phases, mirroring the three claim kinds:

- **Phase A — Universal** (base model on `prepared_data/test/`): discriminative metrics. Fast (~30 s on CPU).
- **Phase B — Deployment** (per-site fine-tune + [test_deployment.py](safecommute/pipeline/test_deployment.py) on held-out ambient + threat wavs). Slow (~10 min CPU per site); `--skip` flag for development; must pass in CI before release.
- **Phase C — Narrative** (before/after deltas cited in the pitch/paper/website). Cross-validates Phase A and B numbers against the marketing-style claims.

### 6.1 What it must verify (each = one assertion with tolerance)

#### Phase A — Universal, base model on `prepared_data/test/`
| # | Claim | Tolerance | Source |
|---|---|---|---|
| 1 | AUC-ROC = 0.804 | ±0.010 | [RESULTS.md:7](RESULTS.md#L7), [README.md:21](README.md#L21) |
| 2 | Accuracy @ thr=0.50 = 0.703 | ±0.010 | [RESULTS.md:8](RESULTS.md#L8) |
| 3 | F1 weighted @ thr=0.50 = 0.716 | ±0.010 | [RESULTS.md:9](RESULTS.md#L9) |
| 4 | Params = 1,829,444 (~1.83M) | ±1% | [RESULTS.md:10](RESULTS.md#L10), [CLAUDE.md](CLAUDE.md), [update.md:29](update.md#L29) |
| 5 | Float32 size = 7.00 MB | ±0.1 MB | [RESULTS.md:11](RESULTS.md#L11) |
| 6 | TPR as_yell = 0.906 | ±0.03 | [RESULTS.md:20](RESULTS.md#L20) |
| 7 | TPR as_screaming = 0.791 | ±0.03 | [RESULTS.md:21](RESULTS.md#L21) |
| 8 | TPR yt_scream = 0.782 | ±0.03 | [RESULTS.md:22](RESULTS.md#L22) |
| 9 | TPR as_shout = 0.647 | ±0.04 | [RESULTS.md:23](RESULTS.md#L23) |
| 10 | FPR yt_metro = 0.351 | ±0.04 | [RESULTS.md:28](RESULTS.md#L28) |
| 11 | FPR as_crowd = 0.579 | ±0.04 | [RESULTS.md:29](RESULTS.md#L29) |
| 12 | FPR as_speech = 0.717 | ±0.04 | [RESULTS.md:30](RESULTS.md#L30) |
| 13 | FPR as_laughter = 0.825 | ±0.04 | [RESULTS.md:31](RESULTS.md#L31) |
| 14 | Base speech-FP headline ≈ 72% | 0.60–0.85 band | [RESULTS.md:35](RESULTS.md#L35), [short.md:98](web/public/demo/short.md#L98), [update.md:160](update.md#L160), [selling.md:5](selling.md#L5) |
| 15 | Base micro threat recall ≈ 82% | ±0.03 | [update.md:114](update.md#L114) |
| 16 | Base overall FPR (matches the 41% leakage claim under its documented configuration) | once the configuration is documented | [update.md:114](update.md#L114) |

*Missing from RESULTS.md but true: `as_gunshot` 89%, `as_explosion` 74%, `as_glass` 80%, `viol_violence` 96% TPR. Add to RESULTS.md; also add rows for them here with the same tolerance.*

#### Phase B — Deployment, per-site fine-tuned model via [test_deployment.py](safecommute/pipeline/test_deployment.py)

Each site is a tuple `(name, ambient_dir, threat_dir)` declared at the top of the script. For each site:
1. 80/20 deterministic split of ambient wavs (SHA256 by filename).
2. Run [finetune.py](safecommute/pipeline/finetune.py) on the 80%.
3. Invoke `test_deployment.py --model models/<site>_model.pth --ambient-dir <held-out 20%> --threat-dir <threat_dir>`.
4. Parse structured results.

| # | Claim | Tolerance | Source |
|---|---|---|---|
| 17 | FP rate ≤ 5% on held-out site ambient | ≤ 0.05, *no margin* | [test_deployment.py:14](safecommute/pipeline/test_deployment.py#L14) (formerly DEPLOY.md:69 — file deleted) |
| 18 | Threat recall ≥ 90% on threat wavs | ≥ 0.88 (3-sample noise band) | [test_deployment.py:11](safecommute/pipeline/test_deployment.py#L11) |
| 19 | Bit-deterministic across repeated inference | zero diff over 10 runs | [test_deployment.py:24](safecommute/pipeline/test_deployment.py#L24) |
| 20 | Silence (RMS < 0.003) classified as safe | 100% | [test_deployment.py:28](safecommute/pipeline/test_deployment.py#L28) |

The paper needs **at least 3 sites** (e.g., metro, bar, train platform) to argue generalization — one site is an anecdote.

#### Phase C — Narrative / marketing figures
These are the load-bearing numbers on the homepage, pitch deck, and paper intro. Each must match a Phase A or B measurement within tolerance.

| # | Claim | Tolerance | Source |
|---|---|---|---|
| 21 | Post-fine-tune speech FP: **71.7% → 4.2%** (10-min fine-tune) | pre: ±0.03 (matches claim 14); post: ≤ 0.08 (doc is specific but measurement varies per ambient dataset) | [update.md:116](update.md#L116), [docs/gamma_prompt.md:85](docs/gamma_prompt.md#L85) |
| 22 | Post-fine-tune threat recall = **~86%** | ±0.03 | [paper.md:34](paper.md#L34), [update.md:122](update.md#L122) |
| 23 | **~50× smaller, ~10× faster** than CNN14/AST | 50× param-ratio must be within ±20%; 10× latency-ratio hard to verify without measuring SOTA models on same hardware — mark *soft* until a SOTA latency row exists | [paper.md:35](paper.md#L35), [docs/gamma_prompt.md:82](docs/gamma_prompt.md#L82) |
| 24 | Gamma ablation: γ=3.0 → AUC 0.856 / 0% hard-neg; γ=0.5 → AUC 0.804 / balanced | γ=0.5 side: ±0.010 (same as claim 1); γ=3.0 side: *documented snapshot* — not re-runnable without retraining | [paper.md:32](paper.md#L32), [update.md:102](update.md#L102) |

#### Latency and size (cross-phase)
| # | Claim | Check | Source |
|---|---|---|---|
| 25 | FP32 CPU mean latency ≤ 15 ms, p99 ≤ 30 ms (deployment gate) | **hard** when `TARGET_HW` env var names the claim's hardware; **soft** otherwise (print measured and hardware id, mark N/A vs spec) | [test_deployment.py:18-19](safecommute/pipeline/test_deployment.py#L18-L19), [safecommute/export.py:23](safecommute/export.py#L23) |
| 26 | **~12 ms** on Ryzen 5, 1 core (marketing) | soft, same pattern | [short.md:44](web/public/demo/short.md#L44), [README.md:20](README.md#L20), [CLAUDE.md](CLAUDE.md) |
| 27 | INT8 ONNX mean latency ≤ 15 ms | soft, same pattern | once 5.2 lands |
| 28 | INT8 AUC degradation ≤ 0.02 vs FP32 | **hard** | once 5.2 lands |
| 29 | Model size: FP32 ≤ 10 MB, INT8 ≤ 6 MB | **hard** | [test_deployment.py:21](safecommute/pipeline/test_deployment.py#L21) |

### 6.2 Hard vs soft
Each check returns `(name, passed, measured, expected, hard, detail)`:
- **Hard**: failing means a doc lies. Exit 1.
- **Soft**: hardware-dependent or aspirational. Print measured, mark "not directly comparable", exit unaffected.

Hard: 1-13, 14 (speech-FP band), 15, 17-22, 25 (on target HW), 28, 29.
Soft: 16 (until the configuration is pinned), 23 (SOTA latency row), 24 (γ=3.0 side), 25 (off target HW), 26, 27.

### 6.3 Reuse, don't re-implement
- [tests/analyze_model.py](tests/analyze_model.py) → claims 1-15.
- [tests/validate_fp_claims.py](tests/validate_fp_claims.py) → claim 14 (Phase A base speech-FP).
- [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py) → claims 17-20 (Phase B, via subprocess or refactor to importable form).
- [tests/measure_latency.py](tests/measure_latency.py) → claims 25-27.
- [safecommute/model.py](safecommute/model.py) — untouched.

Extract shared helpers into `tests/_common.py` so `verify_performance_claims.py` imports rather than duplicates.

### 6.4 Skeleton

```python
"""
tests/verify_performance_claims.py — single authoritative check that every
numerical performance claim in the repo matches the checkpoint and data on
disk.

Exit 0 = all hard claims pass. Exit 1 = at least one hard claim fails.
Soft claims (latency on non-target hardware, aspirational comparisons) print
but never cause failure.
"""

import os, sys
from tests._common import (
    load_base_model, load_test_set, run_inference,
    per_source_metrics, load_or_train_finetuned,
    invoke_test_deployment,
)

SITES = [
    ("metro", "raw_data/youtube_metro", "raw_data/youtube_screams"),
    # ("bar",   "raw_data/bar",           "raw_data/youtube_screams"),
    # ("platform", "raw_data/platform",    "raw_data/youtube_screams"),
]

def check(name, measured, expected, tol, *, hard=True, kind='abs'):
    if kind == 'abs':
        passed = abs(measured - expected) <= tol
    elif kind == 'lte':
        passed = measured <= expected + tol
    elif kind == 'gte':
        passed = measured >= expected - tol
    else:
        raise ValueError(kind)
    return dict(name=name, measured=measured, expected=expected, tol=tol,
                passed=passed, hard=hard)

def phase_a():
    checks = []
    probs, labels, names = run_inference(load_base_model(), load_test_set())
    checks += overall_checks(probs, labels)                # 1-5
    checks += per_source_tpr_checks(probs, labels, names)  # 6-9 (+gunshot/glass/...)
    checks += per_source_fpr_checks(probs, labels, names)  # 10-13
    checks += [speech_fp_headline(probs, labels, names)]   # 14
    checks += [micro_recall_check(probs, labels)]          # 15
    checks += [leakage_claim_check(probs, labels)]         # 16 (soft until pinned)
    return checks

def phase_b(skip=False):
    if skip: return []
    checks = []
    for name, ambient, threats in SITES:
        held_out = split_and_finetune(name, ambient)        # produces models/<name>_model.pth
        result = invoke_test_deployment(
            f"models/{name}_model.pth", held_out, threats)
        checks += deployment_checks(name, result)           # 17-20
    return checks

def phase_c(phase_a_results, phase_b_results):
    checks = []
    # 21: speech FP pre/post — pulls from Phase A (pre) + a Phase-B post-finetune
    # speech-only FPR measurement (new helper).
    # 22: recall ≥ 86% — compare to Phase B recall average.
    # 23: param ratio vs CNN14/AST (static arithmetic); latency ratio soft.
    # 24: gamma ablation — validates current AUC only; γ=3.0 side is a
    #     documented snapshot.
    checks += narrative_checks(phase_a_results, phase_b_results)
    return checks

def latency_and_size():
    checks = []
    checks += latency_checks()                              # 25-27
    checks += size_checks()                                 # 29
    if os.path.exists("models/safecommute_v2_int8.onnx"):
        checks += [int8_auc_check()]                        # 28
    return checks

def main():
    a = phase_a()
    b = phase_b(skip='--skip-phase-b' in sys.argv)
    c = phase_c(a, b)
    ls = latency_and_size()
    checks = a + b + c + ls
    print_report(checks)
    any_hard_failed = any(not c['passed'] and c['hard'] for c in checks)
    sys.exit(1 if any_hard_failed else 0)
```

### 6.5 Sample output (target)

```
SafeCommute AI — performance-claim verification
===============================================

PHASE A  Universal (base model, prepared_data/test/)
  [PASS] AUC-ROC                 0.804  exp 0.804±0.010
  [PASS] Accuracy @ 0.50         0.703  exp 0.703±0.010
  [PASS] F1 weighted @ 0.50      0.716  exp 0.716±0.010
  [PASS] Params                  1,829,444
  [PASS] Size (FP32)             7.00 MB
  [PASS] TPR as_yell             0.906  exp 0.906±0.03
  [PASS] TPR as_screaming        0.791  exp 0.791±0.03
  [PASS] TPR yt_scream           0.782  exp 0.782±0.03
  [PASS] TPR as_shout            0.647  exp 0.647±0.04
  [PASS] FPR yt_metro            0.351  exp 0.351±0.04
  [PASS] FPR as_crowd            0.579  exp 0.579±0.04
  [PASS] FPR as_speech           0.717  exp 0.717±0.04
  [PASS] FPR as_laughter         0.825  exp 0.825±0.04
  [PASS] Speech-FP headline      0.717  band [0.60, 0.85]
  [PASS] Micro threat recall     0.809  exp 0.82±0.03
  [FAIL] Leakage 41% claim       0.342  exp 0.41±0.03   ← doc update needed

PHASE B  Deployment (site: metro)
  [PASS] FP on held-out ambient  0.038  exp ≤ 0.05
  [PASS] Threat recall           0.906  exp ≥ 0.88
  [PASS] Determinism             0 drift over 10 runs
  [PASS] Silence gating          100%

PHASE C  Narrative
  [PASS] Speech FP 71.7%→4.2%    pre=0.717, post=0.061  exp post ≤ 0.08
  [PASS] Post-finetune recall    0.906  exp 0.86±0.03
  [WARN] 50× smaller vs SOTA     ratio=44× (CNN14 80M / 1.83M)  ← within band
  [WARN] 10× faster vs SOTA      no measured SOTA row present
  [PASS] γ=0.5 AUC               0.804  (matches claim 1)
  [WARN] γ=3.0 AUC 0.856         snapshot — not re-runnable

LATENCY / SIZE
  [WARN] Deploy ≤ 15 ms          108.3 ms  on Ryzen 7 7435HS, 8T  (not target HW)
  [WARN] Marketing ~12 ms        108.3 ms  (target is Ryzen 5, 1T — N/A here)
  [PASS] Size FP32 ≤ 10 MB       7.00 MB
  [PASS] Size INT8 ≤ 6 MB        2.10 MB  (after 5.2)

Result: 17 PASS, 1 FAIL, 5 WARN  →  exit 1  (leakage mismatch)
```

This is one line in the paper's Reproducibility section:
*"All reported performance numbers are verified by `tests/verify_performance_claims.py`, which runs in ~12 minutes on CPU and must exit 0."*

---

## 7. Paper-readiness checklist

Do not submit before every box ticks:

- [ ] **5.1** ONNX export produces a single self-contained file.
- [ ] **5.2** Static INT8 INT8 ONNX exists; accuracy within 2pp of FP32.
- [ ] **5.3** Conv+BN folding applied; FP32 AUC unchanged.
- [ ] **5.0** `tests/validate_fp_claims.py` Phase C measures site-held-out ambient, not the universal test set.
- [ ] **5.5** Pi 4 and Pi 5 measurements added to RESULTS.md with hardware details.
- [ ] **5.6** End-to-end latency (preprocess + model) reported alongside model-only.
- [ ] **5.7** The 41% leakage figure in [update.md:114](update.md#L114) matches a measured configuration or is updated.
- [ ] Section 6 `tests/verify_performance_claims.py` exists; Phases A+B+C all exit 0 on at least 3 sites.
- [ ] Per-source FPR table in [RESULTS.md](RESULTS.md) expanded to all 16 safe sources ([analyze_model.py](tests/analyze_model.py) output pasted in).
- [ ] Gamma ablation (claim 24 γ=3.0 side) either (a) has a saved `models/safecommute_v2_gamma3.pth` so the AUC can be re-measured, or (b) is labeled "historical snapshot" in the paper.
- [ ] SOTA comparison (claim 23) has CNN14 and AST measured on the same hardware/setup.
- [ ] Every numerical claim in the paper draft has a matching row in `verify_performance_claims.py`.
- [ ] Hardware details (CPU model, thread count, governor, ambient temp, BLAS backend, PyTorch/ONNX versions) reported for every latency number.

## 8. Website-readiness checklist

- [ ] **5.4** Demo zip contains INT8 ONNX + `infer.py` + `PCEN.py` + README — not the raw `.pth`.
- [ ] Homepage numbers (12 ms, 5%, 4.2%, 86%) all have a green row in `verify_performance_claims.py`; re-measure on the cited hardware or change the copy.
- [ ] Fine-tuning flow (user uploads `.wav` → [finetune.py](safecommute/pipeline/finetune.py) → returns `.onnx` + `thresholds.json`) tested end-to-end on the uploaded-ambient → personalized-INT8 path.
- [ ] The returned personalized model passes `tests/validate_fp_claims.py --skip-finetune` on held-out audio from the user's upload (leave-one-file-out check).
- [ ] Privacy claim verified: no raw audio leaves the user's session; only PCEN spectrograms are persisted. Audit the upload handler.

---

## 9. How to work through this

Suggested order (lowest risk first, quickest paper-readiness):

1. **5.1** (fix ONNX export) — unblocks everything downstream.
2. **Section 6** (write `verify_performance_claims.py`, Phases A + C only) — establishes the bar, surfaces what is actually failing.
3. **5.0** (fix validate_fp_claims Phase C) — makes the fine-tuning claim checkable.
4. **5.3** (Conv+BN fuse) — free win, zero regression risk.
5. **5.2** (static INT8 PTQ) — the one real latency lever.
6. **Section 6 Phase B** (run per-site fine-tunes + `test_deployment.py`) — validates the headline 5% / 4.2% / 86% claims.
7. **5.4** (rebuild demo zip) — changes what ships publicly.
8. **5.6** (end-to-end latency) — clarifies the paper number.
9. **5.7** (reconcile leakage figure) — unblocks paper submission.
10. **5.5** (Pi benchmarks) — requires hardware, do last once rest is stable.

Each step lands as its own commit with a fresh `verify_performance_claims.py` run attached to the PR description. If a change breaks a claim, update the claim to the new honest number before merging — do not quietly keep an unverified figure in the docs.

---

## 10. Sellability Roadmap (Product-Side)

§5–§9 cover the paper. This section covers the paid web product. Current commercial state: a beautiful Bauhaus shell with real Supabase auth and real Stripe Checkout webhooks, sitting on top of **zero working backend execution**. Until §10.1 lands, no paying customer can complete a single run.

The pricing model itself is sound and honest — one-time payments framed as "give away the model, sell the calibration" ([`selling.md`](selling.md)), per-site unlock as a moat, per-run credits as a natural re-calibration revenue stream (mic swaps, seasonal change). The problem is execution, not strategy.

### 10.1 Build the fine-tune worker — **#1 launch blocker**

**Why**: jobs are queued into Supabase `finetune_jobs` with `status='queued'` but nothing consumes them. The paid product cannot complete a run.

**How**: a small Python service (Railway / Fly / Modal — anywhere that can run PyTorch) that:
- polls `finetune_jobs WHERE status='queued' ORDER BY created_at LIMIT 1` every 30 s, marking selected rows `status='running'` atomically,
- downloads the user's clips from the `audio-uploads` Storage bucket via service-role key,
- shells out to [`safecommute/pipeline/finetune.py`](safecommute/pipeline/finetune.py) with `--ambient-dir <tmp>` and `--environment <site_id>`,
- uploads the resulting `<job_id>_model.pth` (or INT8 ONNX once §5.2 lands) and `thresholds.json` back to a private bucket,
- updates the row to `status='succeeded'` with the artefact URLs, or `status='failed'` with the stderr tail.

**Verify**: a fresh test user can sign in → pay €23 → upload a clip → the job transitions queued → running → succeeded within ~10 min, with the artefact downloadable.

### 10.2 Implement model-download route

**Why**: even a succeeded job is unusable without a download endpoint. The dashboard shows a download button conditional on `status === 'succeeded'` but the route doesn't exist.

**How**: add `GET /api/finetune/[id]/download` in [`web/app/api/finetune/`](web/app/api/finetune/) that:
- validates the request user owns the job (RLS already enforces this at the DB layer; double-check at the route),
- returns a signed Storage URL (60 s TTL) for the model + thresholds.

**Verify**: `curl` with a valid session cookie returns a 302 to a signed Supabase Storage URL; downloads a working artefact.

### 10.3 Fix the privacy claim — currently structurally false

**Why**: the web copy ([`web/components/privacy-section.tsx`](web/components/privacy-section.tsx), [`web/components/hero.tsx`](web/components/hero.tsx)) says "no audio leaves the device" / "edge-only". This is **true for [`safecommute/pipeline/inference.py`](safecommute/pipeline/inference.py)** (RAM-only rolling buffer, PCEN never inverted, nothing on disk). It is **false for the paid fine-tune flow** — clips upload to Supabase Storage as raw `.wav` before PCEN is applied server-side by the worker (§10.1).

**How** (pick one):
- (a) **Apply PCEN client-side** before upload: a WebAssembly PCEN module in the browser, only the resulting spectrogram tensor uploads. This preserves the hard claim. Larger lift (~1–2 weeks) but it is the differentiator.
- (b) **Encrypt the upload bucket end-to-end** with a per-job key, delete the raw clip the moment training starts, and update the copy to "audio is processed in an encrypted, ephemeral pipeline and never persisted". Smaller lift (~1–2 days) but the claim is weaker.

Soften the copy first (the cheap honest thing), then implement (a) when there is bandwidth.

**Verify**: a network capture of a fine-tune session shows either no raw audio leaving the browser (option a) or shows the bucket auto-deletes within seconds of training start (option b). Web copy matches whichever is true.

### 10.4 Ship a minimal SDK + integration guide

**Why**: customers who download the demo `.pth` (or a fine-tuned artefact) have nowhere to go from "I have the file" to "it runs in my product". `web/public/demo/short.md` is a quickstart, not a deployment guide.

**How**: a thin `safecommute-sdk` Python package with three responsibilities:
- load `.pth` or `.onnx` + `thresholds.json`,
- expose a `classify(samples_int16: np.ndarray) -> {prob, label, smoothed}` matching [`safecommute/pipeline/inference.py`](safecommute/pipeline/inference.py)'s loop semantics (rolling 3 s buffer, 1 s stride, 4-pred temporal smoothing, dual amber/red thresholds, RMS energy gate, F0-aware threshold raising on speech),
- provide a `serve()` CLI that turns the SDK into a microphone listener for production-style demos.

Plus a `docs/sdk/integration.md` page on the website with copy-paste snippets for: Linux microphone (ALSA/PortAudio), MQTT/HTTP egress for alerts, log-only dry-run mode, threshold tuning UI hooks.

**Verify**: a fresh-venv user runs `pip install safecommute-sdk`, loads the demo model, points it at a `.wav`, gets a sane prediction.

### 10.5 Rename Stripe products + add Customer Portal

**Why**: the underlying Stripe products are named "Subscription" while the modal copy correctly says "one-time payment". This invites chargebacks and confused customers.

**How**: in the Stripe dashboard, rename SKUs to "Per-run credit (one-time)" and "Site unlock (one-time, lifetime)". In [`web/lib/stripe.ts`](web/lib/stripe.ts), add a `/api/stripe/portal` route that returns a Customer Portal session URL so customers can manage payment methods, see receipts, and request refunds without emailing support.

**Verify**: the dashboard "Manage billing" link opens a real Stripe Customer Portal session.

### 10.6 Compliance docs (required for transit / schools / elder care)

**Why**: the entire vertical list in [`web/components/verticals-grid.tsx`](web/components/verticals-grid.tsx) (transit, retail, schools, elder care, industrial) is regulated. None will sign a contract without these docs.

**How**: write minimal versions of:
- `docs/security.md` — auth model, data flow diagram, encryption-at-rest/in-transit, sub-processor list (Supabase, Stripe, Vercel, the worker host).
- `docs/dpa.md` — GDPR Data Processing Agreement template (use the EU SCC 2021/914 module two as a base).
- `docs/retention.md` — what is kept (job metadata: forever; raw uploads: deleted on training start per §10.3; trained models: indefinitely under the user's account).
- Footer link in [`web/components/site-footer.tsx`](web/components/site-footer.tsx).

**Verify**: a regulated buyer can read all four docs in <15 min and the answers are unambiguous.

### 10.7 Add per-user job concurrency / rate limits

**Why**: one paying user can otherwise queue 1000 jobs and starve everyone else's worker capacity.

**How**: in the worker (§10.1) and at the `/api/finetune/trigger` route, enforce: max 1 running job per user, max 5 queued, configurable burst budget per Enterprise customer.

**Verify**: a synthetic load test that submits 20 jobs from one user keeps 1 running + 5 queued and 429s the rest with a sensible "try again when one finishes" message.

### 10.8 Build 1–2 reference case studies

**Why**: cold sales calls without case studies bounce. The §6 Phase B per-site validation doubles as case-study material if the sites are real.

**How**: pick two real environments you have access to (a café, your own apartment, the library — anywhere with 30+ min of recordable ambient and a few real shouts). Run §10.1 end-to-end. Publish the before/after FP+TPR numbers as `web/app/case-studies/[slug]/page.tsx` with the recording setup documented for credibility.

**Verify**: a `/case-studies/cafe` page exists with measured numbers traceable to a `models/<site>_model.pth` checkpoint and an `analyze_model.py` JSON dump committed to the repo (or attached to the case study).

### 10.9 Fix the latency claim on the web hero

**Why**: [`web/components/hero.tsx`](web/components/hero.tsx) shows "12 ms" as a top-line stat. A technical buyer who clones the repo, runs `tests/measure_latency.py`, and gets 108 ms walks away. This is risk-doubling: the same unverified number on the README and the hero.

**How**: re-measure on the actual hardware the page targets (Ryzen 5, 1 thread — or pick a representative budget x86 machine you can demo on), update the hero, and stop quoting 12 ms anywhere unless it reproduces.

**Verify**: hero number matches a fresh `tests/measure_latency.py` row on disclosed hardware.

### 10.10 Add `web/.env.example`

**Why**: setting up `/web/` on a new machine currently requires walking through [DEPLOY_WEB.md §3-4](DEPLOY_WEB.md) by hand. Reproducibility gap.

**How**: commit a `web/.env.example` with placeholder values for every required env var (Supabase URL, anon key, service-role key, Stripe public key, Stripe secret, Stripe webhook secret, base URL).

**Verify**: a fresh contributor can `cp .env.example .env.local`, fill in values, and `npm run dev` works first try.

---

## 11. One-Week Sprint (Highest-Leverage First)

If only five things land this week, do them in this order:

1. **Run [`safecommute/pipeline/test_deployment.py`](safecommute/pipeline/test_deployment.py) on one real site.** Record 30+ min of metro/bar/café ambient yourself, fine-tune, measure FP+recall. This single number unblocks ~70% of the paper (item §5.0 + §6 Phase B) AND ~70% of the sales pitch (item §10.8).
2. **Replace the 41% literal in [`scripts/generate_pitch_figures.py:267-268`](scripts/generate_pitch_figures.py#L267-L268) with measured numbers** read from the actual model. Regenerate `confusion_matrix.png` and `finetune_impact.png` from real eval. (§5.7.)
3. **Re-run [`tests/measure_latency.py`](tests/measure_latency.py) and put the actual numbers + hardware in [`README.md`](README.md), [`RESULTS.md`](RESULTS.md), and [`web/components/hero.tsx`](web/components/hero.tsx).** Stop quoting 12 ms anywhere unless it reproduces. (§10.9.)
4. **Spike the worker (§10.1).** Even a single-machine Python script polling Supabase every 30 s and shelling out to `finetune.py` is enough to prove the loop closes end-to-end. Productionise later.
5. **Soften the privacy copy (§10.3 option b)** until PCEN-on-device is implemented. Replace "no audio leaves the device" with "audio is processed in an encrypted, ephemeral pipeline and never persisted" on every web component that says it.

After these five, the next batch of two-week work is: §10.2 (download route), §10.4 (SDK), §5.0 (fix Phase C), §5.1 (ONNX export), §6 (verify_performance_claims.py Phase A+C). Then §10.3 option a (WASM PCEN), §10.6 (compliance docs), §5.5 (Pi benchmarks), and the rest of §6 (Phase B on 3 sites) close out the 6–8 week window to defensible paper + launchable product.

---

## 12. What Actually Landed (end-of-session audit, 2026-04-21)

### §5 + §6 (paper-readiness work) — mostly done

| Item | Status | Evidence |
|---|---|---|
| 5.0 Fix `tests/validate_fp_claims.py` Phase C to measure held-out site ambient | **Done** | Rewritten; emits `tests/reports/phase_b_<env>.json`. Measured the gap (see below). |
| 5.1 Single-file ONNX (no sidecar) | **Done** | [safecommute/export.py:101-139](safecommute/export.py); `models/safecommute_v2.onnx` = 6.99 MB, no `.data` sidecar, parity 2.4e-7 vs PyTorch. |
| 5.2 Static INT8 PTQ on Conv | **Done** | New [safecommute/export_quantized.py](safecommute/export_quantized.py); `models/safecommute_v2_int8.onnx` = 3.72 MB, AUC Δ = 0.002, mean latency 2.86 ms @ 8T. |
| 5.3 Conv+BN folding | **Done** as part of 5.2 | ORT graph optimizer handles the fusion during `quant_pre_process`; `models/safecommute_v2_fused.onnx` = 6.99 MB, FP32 AUC unchanged. |
| 5.4 Demo zip PyTorch-free | **Done** | [web/scripts/build-demo-bundle.mjs](web/scripts/build-demo-bundle.mjs) rewritten; zip = 3.41 MB (INT8 ONNX + `infer.py` + `feature_stats.json` + README). No `torch` dep; `librosa + scipy + numpy + onnxruntime` only. |
| 5.5 Raspberry Pi benchmark | **Deferred** | Hardware not on hand. |
| 5.6 End-to-end latency | **Done** | [tests/measure_e2e_latency.py](tests/measure_e2e_latency.py); reports preprocess + model + total with HW disclosure. Result: preprocess dominates (~16 ms) on x86; model-only 2.8 ms INT8. |
| 5.7 Reconcile 41 % leakage | **Done** | [scripts/generate_pitch_figures.py](scripts/generate_pitch_figures.py) now reads [tests/reports/figures_source.json](tests/reports/figures_source.json). Zero hand-typed literals. Measured value is 34.2 % FPR @ thr=0.5. |
| 6 `tests/verify_performance_claims.py` (Phase A + C + latency + size + INT8 parity) | **Done** | 41 rows, JSON report, figures-source emit. Exit 1 currently because 4 load-bearing post-fine-tune claims (5 %, 4.2 %, 86 %) do not reproduce — **by design**; this is the honest gate. |

### §7 Paper-readiness checklist — now auditable

- [x] 5.1, 5.2, 5.3, 5.0, 5.6, 5.7 complete.
- [x] `verify_performance_claims.py` exists.
- [x] Per-source FPR table expanded to all 16 safe sources in [RESULTS.md](RESULTS.md).
- [x] γ=3.0 ablation row marked *historical snapshot* in every figure + paper (checkpoint not preserved; retrain queued as §7 gap item 1 in [paper.md](paper.md)).
- [ ] SOTA comparison still soft-labeled (no measured CNN14/AST row on same hardware; see [paper.md](paper.md) §4 gap item 7).
- [x] Every numerical claim in [paper.md](paper.md) has a matching row in the verifier JSON.
- [x] Hardware disclosure attached to every latency number via `tests/_common.hw_disclosure()`.

### §10 Product / web backend — out of scope for this session

Per user decision (scope = paper + §5.4 demo bundle only), the fine-tune worker, download route, privacy-copy fix, SDK, Stripe rename, compliance docs, rate limits, case studies, and `.env.example` (§10.1–§10.10) were **not attempted**. They remain the launch blockers before a paying customer can complete a flow. Current state: as documented in §10 above.

### §11 One-week sprint — what landed

1. [x] Ran `safecommute/pipeline/test_deployment.py` on one real site (metro). **Result**: FP 38.2 %, recall 89.5 % — gate not met.
2. [x] Replaced the 41 % literal. Confusion-matrix figures regenerate from [tests/reports/figures_source.json](tests/reports/figures_source.json).
3. [x] Re-measured latency; [README.md](README.md) / [RESULTS.md](RESULTS.md) carry the hardware-disclosed table. 12 ms kept only as historical footnote (user decision).
4. [ ] Worker spike (§10.1) — out of scope.
5. [ ] Privacy copy soft-fix (§10.3 option b) — out of scope; [paper.md](paper.md) §5 lists the exact web components to edit.

### New — §13 Tweak sweep findings (not originally in this plan)

The planned work ran end-to-end and the deployment gate failed on the one real site (§12 row 1). Per the user's rigor mandate ("never fabricate numbers; try every architecture-preserving lever before acknowledging the limit"), we swept 5 architectural-preserving tweaks on the same metro fine-tune. **All measured on the 34 truly-held-out `youtube_metro_quarantine` wavs.**

| Tweak | Recipe | Default-thr FP / recall | Best FP ≤ 5 % point (thr / FP / recall) |
|---|---|---|---|
| 0 | default (keep 0.5, 10 ep, frozen) | 38.2 % / 89.5 % | 0.77 / 2.9 % / 40.4 % |
| 1 | keep 0.2, 15 ep, frozen | **29.4 %** / 89.5 % | 0.73 / 0.0 % / 35.1 % |
| 2 | keep 0.3, 15 ep, **unfreeze + warmup 3** | 38.2 % / 87.7 % | 0.74 / 2.9 % / 43.9 % |
| 3 | keep 0.1, 20 ep, frozen | **29.4 %** / 87.7 % | 0.71 / 2.9 % / **49.1 %** |
| 4 | threshold-only recal on default | (n/a) | 0.77 / 2.9 % / 40.4 % |

Key findings:
1. **FP plateau at ≈ 29.4 %** across keep_safe ∈ {0.1, 0.2}; more aggressive re-weighting hits a ceiling.
2. **Unfreezing the CNN is actively worse** (tweak 2 regresses to the default 38.2 %). Adaptation is a decision-surface problem, not a feature-extraction problem.
3. **Best FP ≤ 5 % operating point: tweak 3, recall 49.1 %** (+8.7 pts absolute vs default at the same FP). This is the headline positive result.
4. **Gate unreachable via training-side tweaks alone** on the 58-clip fine-tune pool. Remaining architecture-preserving levers (not yet attempted; see [paper.md](paper.md) §7):
   - Calibrate `low_fpr` on *site* ambient during fine-tune, not on the universal test set (two-line patch in [safecommute/pipeline/finetune.py:390-394](safecommute/pipeline/finetune.py)).
   - Temporal-majority aggregation: require ≥ 2 consecutive over-threshold windows before flagging a FP ([safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py) `test_false_positive`).
   - ≥ 5× more field-recorded per-site ambient (58 clips → 300+).

### Where [paper.md](paper.md) goes next

The paper is *workshop-submittable* in current form as a deployment-gap case study with the plateau + unfreeze negative finding + tweak-3 positive finding. Before submission, the items in [paper.md §4 Gap list](paper.md) are the critical path: re-enable γ=3.0 (retrain once), measure on ≥ 2 sites, implement the two remaining levers above. Estimate: 3–4 weeks of focused work to a defensible workshop short paper.

---

## 14. 2026-04-22 session landing — Phase 1, 2, 4 done

The 2026-04-21 session closed by listing three blockers for a workshop-submittable paper and a launchable paid product. All three are resolved this session; only one category (field recordings on ≥ 2 more sites) remains and requires user action, not code.

### Phase 1 — architecture-preserving levers close the FP gate on n=1 site

| Lever | Where it lives | Effect (measured 2026-04-22, metro) |
|---|---|---|
| Temporal-majority aggregation (k=2) | `fires()` in [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py); `--majority-k` CLI flag; mirror in [safecommute/pipeline/inference.py](safecommute/pipeline/inference.py) (`MAJORITY_K=2`) | metro_tweak3 at pre-lever `low_fpr` 0.601 + k=2 → **FP 0.0 %** / recall 78.9 % (was 31.6 % / 87.7 % at k=1 on the same split). Pre-lever → post-lever: −31.6 pts FP for −8.8 pts recall. |
| Site-ambient threshold recalibration (`low_fpr_site`) | `--calibration-ambient-dir` + `--calibration-majority-k` in [safecommute/pipeline/finetune.py](safecommute/pipeline/finetune.py); the CLI now emits `low_fpr_site` to the thresholds JSON; [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py) loader prefers it | At 15-cal-wav budget the lever **over-tightens** (6.7 % cal resolution pushes sweep to 0-FP thresholds, costing −40 pts recall without extra FP margin). Architecturally sound; needs ≥ 30 cal wavs to help. See paper.md §1.4 tweak 5. |
| Deterministic 50/50 held-out split | [tests/eval_metro_with_levers.py](tests/eval_metro_with_levers.py) (site-agnostic: `--site <name> --held-out-dir … --checkpoints-file …`); [tests/pick_best_phase_b.py](tests/pick_best_phase_b.py) picks the best FP ≤ 5 % operating point | Sha256 salt encodes the site name — runs are idempotent, two sites don't alias onto the same split. |

Report artefacts: [tests/reports/metro_lever_sweep.json](tests/reports/metro_lever_sweep.json) (full 4 ckpts × 3 k × 2 threshold-choice matrix), [tests/reports/phase_b_metro.json](tests/reports/phase_b_metro.json) (rewritten to carry the honest-best operating point), [tests/reports/tweak_finetune.json](tests/reports/tweak_finetune.json) (tweak5 appended).

### Phase 4 — paper completeness

| Item | Status | Evidence |
|---|---|---|
| γ=3.0 retrain | **Done** | `models/safecommute_v2_gamma3.pth` (7 MB, SHA in [tests/reports/artefacts.sha256](tests/reports/artefacts.sha256); 19 epochs early-stop, GPU). AUC **0.791** (not 0.856 — material correction; the γ-collapse pattern is reproduced in per-source hard-neg accuracy 3.9 %, threat TPR 95.9 %). |
| SOTA baseline on same hardware | **Done** | [tests/measure_sota_baselines.py](tests/measure_sota_baselines.py); [tests/reports/baselines.json](tests/reports/baselines.json) (8T) + [tests/reports/baselines_1t.json](tests/reports/baselines_1t.json). PANNs CNN14: 81.8 M params, 312.3 MB FP32, 100.1 ms median @ 8T, 250.5 ms @ 1T. Measured ratios vs SafeCommute INT8 ONNX: **44.7× smaller / 22.7× faster @ 8T / 13.9× faster @ 1T**. Original "~50× / ~10×" paper claims now hardware-disclosed and exceeded. |
| Verifier reads γ=3.0 + baselines live | **Done** | [tests/verify_performance_claims.py](tests/verify_performance_claims.py): γ=3.0 Phase A row loads the live checkpoint and reports real AUC + hard-neg acc on speech+laughter+crowd; Phase C `"~10× faster"` row reads `baselines.json` and is one-sided (`≥ 10×`). |
| paper.md propagation | **Done** | §0 TL;DR (both tables + headline), §1.3 (Phase B recipe + new protocol), §1.4 (tweak sweep table including tweak 5), §2.2 #1 (γ-collapse reframing), §2.3 (SOTA reframing with measured numbers), §3.2 (ablation table — new), §3.3 (SOTA comparison table — new), §3.5 (experiment list), §4.3 (γ-collapse discussion), §4.5 (operator implications), §5 limitations, §6 conclusion reproducibility command, §4 gap list, §7 this-week list. |
| ≥ 2 more sites for Phase B | **Pending user action** | Infrastructure ready: [tests/eval_metro_with_levers.py](tests/eval_metro_with_levers.py) accepts `--site <name> --held-out-dir raw_data/<site>_heldout --checkpoints-file <site>.json`. One-liner workflow documented in paper.md §4 item 3. |

### Phase 2 — paying customer can complete a flow

| Piece | Status | Where |
|---|---|---|
| Migration for `models-deliverable` bucket + `worker_logs` table | **Done** | [web/supabase/migrations/0002_models_deliverable_bucket.sql](web/supabase/migrations/0002_models_deliverable_bucket.sql). Private bucket with self-read RLS scoped to `{owner}/…`; `worker_logs` has `job_id` FK + level + message + traceback, RLS self-read through the parent job. |
| Fine-tune worker | **Done** | [worker/main.py](worker/main.py) (poll loop, signal handling), [worker/job.py](worker/job.py) (end-to-end: download → 80/20 split → tweak-3 fine-tune + calibration → INT8 ONNX → deployment report → upload 3 artefacts → mark succeeded → **wipe source clips**), [worker/export.py](worker/export.py) (per-job path-safe INT8 wrapper — reuses `safecommute.export_quantized` helpers without touching the base artefacts), [worker/supabase_client.py](worker/supabase_client.py). |
| Download route | **Done** | [web/app/api/finetune/[id]/download/route.ts](web/app/api/finetune/[id]/download/route.ts) — GET `?file=model|thresholds|report`, 60-second signed URLs, 302 redirect (or `?format=json` for programmatic). |
| Dashboard 3 download buttons | **Done** | [web/components/dashboard/job-card.tsx](web/components/dashboard/job-card.tsx): `Model .onnx ↓`, `Thresholds ↓`, `Deployment report ↓`. |
| Privacy fix (§10.3 option b) | **Done** | Worker deletes raw clips from `audio-uploads` on success — only non-invertible PCEN-derived weights persist. [web/components/privacy-section.tsx](web/components/privacy-section.tsx) rewritten to match: inference still "no audio leaves device" (structural); fine-tune now "ephemeral bucket + wiped on success". Client-side WASM PCEN (§10.3 option a) remains roadmap. |
| `.env.example` (§10.10) | **Done** | [web/.env.example](web/.env.example) + [worker/.env.example](worker/.env.example) cover every variable with inline comments. `.gitignore` updated to allow the `web/.env.example` template through the broader `web/.env*` rule. |
| Systemd unit (self-hosted v1 deploy target per plan) | **Done** | [systemd/safecommute-worker.service](systemd/safecommute-worker.service), install + linger instructions in [worker/README.md](worker/README.md). |

Verification done autonomously: `tsc --noEmit` exit 0 on the full `web/` tree after the new route + job-card + privacy-section changes; `python -m worker` imports resolve; splitter edge cases (3 clips, 100 clips) produce valid buckets. End-to-end requires a live Supabase project + Stripe test mode — untested but every code path compiles and the protocol matches the 0001 + 0002 schema.

### §10 items still open (not blockers for a first paying customer, but for enterprise sales)

- §10.4 SDK + integration guide — not built. Customers download `model.onnx` + `thresholds.json` + `deployment_report.json` and follow the demo bundle (`web/public/demo/`) for now.
- §10.5 Stripe Customer Portal — not wired (rename deferred since it's pure copy).
- §10.6 Compliance docs (DPA, security, retention) — not written.
- §10.7 Rate limits — not wired (`WORKER_CONCURRENCY=1` and the `--majority-k` recipe are the current backpressure).
- §10.8 Reference case studies — pending ≥ 2 more sites (Phase 4.1).
- §10.9 Hero latency claim rewording — the `/web/components/hero.tsx` still shows the historical "12 ms" disc with a footnote; full rework tied to the marketing pass in Phase 3.

### Honest verifier state post-2026-04-22

`tests/verify_performance_claims.py`: 34 PASS / 6 FAIL / 1 WARN → exit 1. Every FAIL is concrete:

- **B:metro recall ≥ 88 %** — 78.9 % measured (9 pts short; closes at ≥ 30 cal wavs or a learned aggregator)
- **B:metro speech-FP post ≤ 10 %** — 16.9 % measured
- **C:Speech FP post ≤ 10 %** — mirrors Phase B speech-FP
- **C:Post-FT recall ~86 %** — mirrors Phase B recall
- **γ=3.0 AUC 0.856** — measured 0.791 (material correction, paper reframes)
- **C:γ=3.0 AUC 0.856** — mirrors Phase A

No silent contradictions remain between the verifier and paper.md / README.md / RESULTS.md / tests/reports/SUMMARY.md.

### Where this document goes next

All sections §5, §6, §7, §10 (partially) are now done. §8 Website-readiness and Phase 3 (marketing-copy hygiene across the `web/` surfaces) are the next chunk — not attempted this session. Per the plan priority ("paper-first, product-second"), those wait until ≥ 2 more sites are recorded.

---

## 15. 2026-04-27 production landing — flow live in Stripe test mode

The 2026-04-22 session shipped all the code; this session put it into
production and verified the full loop on a real device.

| Layer | What happened on 2026-04-27 |
|---|---|
| Migration 0002 deploy | Applied to the production Supabase project; bucket `models-deliverable` and `worker_logs` table verified via SQL editor. Initial paste failed silently (non-standard MIME types in the bucket allow-list); fix was to trim the allow-list to `application/octet-stream` + `application/json` only and re-run the storage-bucket section idempotently. |
| Worker startup | `safecommute-worker.service` user-systemd unit installed + lingered. Two startup blockers diagnosed in this session: (1) `EnvironmentFile=` does NOT strip inline `# …` comments from `KEY=VALUE` lines — fixed by introducing [worker/env.py](worker/env.py) (defensive `env_int / env_float / env_str` helpers) and rewriting [worker/.env.example](worker/.env.example) to put all comments on their own lines; (2) iPhone clip MP4 uploads bypassed the `audio/*` MIME allow-list and the worker silently exited 0 because [safecommute/pipeline/finetune.py](safecommute/pipeline/finetune.py) only globbed `*.wav` — fixed by extending the bucket allow-list to `video/mp4|quicktime|webm` (patched into [web/supabase/migrations/0001_init.sql](web/supabase/migrations/0001_init.sql) for fresh deploys), loosening the finetune CLI to a tuple of `('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.mp4', '.mov', '.webm')`, and exiting non-zero when `n_ambient == 0` so the worker now propagates a real cause. |
| Vercel production deploy | New download route + 3 dashboard buttons + reframed privacy section pushed live via `cd web && vercel --prod`. Repo-root `vercel --prod` was attempted first and tripped the 10 MB tarball limit (uploads `models/`, `raw_data/`, `prepared_data/`, …); fix is documented in [DEPLOYMENT_NEXT_STEPS.md §3](DEPLOYMENT_NEXT_STEPS.md) and [DEPLOY_WEB.md §4.3](DEPLOY_WEB.md). |
| Vercel ↔ GitHub auto-deploy | **Found broken**: three pushes to `main` between 2026-04-22 and 2026-04-27 produced no Vercel builds (production stayed at the 2026-04-21 deployment until the manual `vercel --prod` ran). Workaround: manual deploy from `web/`. Permanent fix: reconnect GitHub via Vercel → Project Settings → Git. |
| End-to-end verification | iPhone Safari → upload 3 MP4 voice memos → Stripe test card 4242… → 1 credit → "Run fine-tune" → worker claimed within 15 s → succeeded after ~12 min of CPU fine-tune → 3 download buttons resolved → `.onnx` (3.7 MB), `thresholds.json`, `deployment_report.json` all downloaded → source clips wiped from `audio-uploads/`. Full privacy-fix path verified on a real row. |

### What's still pending (2026-04-27)

- **Stripe live-mode flip** — first real paying customer is gated only on this. Process documented in [DEPLOYMENT_NEXT_STEPS.md §5](DEPLOYMENT_NEXT_STEPS.md).
- **Marketing-copy hygiene (Phase 3, §10.9)** — the hero still reads "12 ms"; full pass scoped to [DEPLOYMENT_NEXT_STEPS.md §6](DEPLOYMENT_NEXT_STEPS.md).
- **GitHub → Vercel auto-deploy reconnection** — operational, not architectural.
- **n=1 → n≥3 sites** — only path to a defensible workshop submission. User-side field-recording is the long-pole.
- **24/7 worker uptime cost** — running on the Ryzen dev box is free but not portable. Migration to a free-tier or paid VPS deferred until the volume justifies HA; brainstorm paths queued (Oracle Always Free / Cloud Run on-demand refactor / paid VPS).

### Worker resilience follow-up (queued)

Observed once on 2026-04-27: a transient DNS `NameResolutionError` between the worker host and Supabase mid-job — the upload to the `models-deliverable` bucket succeeded but the final `mark_job_succeeded` PATCH failed, leaving a row in `status='running'` even though the artefacts were already on the bucket. Recovery was a manual re-flip of the row. Follow-up: add a small retry budget around the final two `supabase` calls in [worker/job.py](worker/job.py) so transient failures don't strand jobs. Not blocking — the failure mode is rare and the data path is correct.
