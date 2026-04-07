# SafeCommute AI — Sequential Build Prompts (v2)

## Architecture & Data Strategy

**Model**: SafeCommuteCNN — CNN6 + SE + GRU + multi-scale pooling (1.83M params, ~7ms CPU inference). DO NOT CHANGE.

**Data philosophy**: Threat sounds are universal (a scream is a scream). "Safe" sounds are deployment-specific (metro ambient ≠ bar ambient). The model is trained in two stages:

1. **Base model** — trained on universal threat sounds (Layer 1) + universal hard negatives (Layer 2). This is the general-purpose checkpoint that ships.
2. **Fine-tuned model** — the base model fine-tuned with deployment-specific ambient audio (Layer 3). This adapts to a specific environment (metro, bar, bus) in minutes.

**Layer 1 — Shared threat sounds (unsafe, label=1):**
| Source | What | Why |
|--------|------|-----|
| AudioSet: Screaming `/m/03qc9zr` | Real screams from YouTube | Gold-standard threat signal |
| AudioSet: Shout `/m/07sr1lc` | Shouting | Escalation precursor |
| AudioSet: Yell `/m/07r660_` | Yelling | Escalation precursor |
| AudioSet: Gunshot `/m/032s66` | Gunfire | Unambiguous threat |
| AudioSet: Explosion `/m/014zdl` | Explosions | Unambiguous threat |
| AudioSet: Glass breaking `/m/07q0yl5` | Glass shattering | Physical violence indicator |
| YouTube screams (existing) | Real public screams | 97.3% accuracy in v1, proven |
| Violence dataset (existing) | Real altercation audio | Proven source |

**Layer 2 — Shared hard negatives (safe, label=0):**
| Source | What | Why |
|--------|------|-----|
| AudioSet: Laughter `/m/01j3sz` | Laughter | Loud, non-threatening — key confound |
| AudioSet: Crowd `/m/03qtwd` | Crowd noise | Dense human sound, not a threat |
| AudioSet: Speech `/m/09x0r` | General speech | Baseline human vocalization |
| AudioSet: Music `/m/04rlf` | Music | Loud, non-threatening |
| AudioSet: Applause `/m/0fx80y` | Clapping/applause | Sudden burst, non-threatening |
| AudioSet: Cheering `/m/03w41f` | Cheering | Loud crowd, non-threatening |
| AudioSet: Singing `/m/015lz1` | Singing | Vocal energy without threat |
| ESC-50 (existing) | Environmental sounds | Ambient baseline |
| UrbanSound8K (existing) | Urban sounds | Ambient baseline |

**Layer 3 — Deployment-specific ambient (safe, label=0, added during fine-tuning):**
| Deployment | Data needed | How to get it |
|------------|-------------|---------------|
| Metro | Train noise, announcements, platform crowd, door chimes | YouTube metro ride compilations (already have 58 files) + record your own |
| Bar/nightlife | DJ music, crowd chatter, clinking, loud conversation | Record 2-3 hours in venues |
| Bus | Engine drone, braking, passenger chatter, traffic | YouTube bus ride compilations + record |

**EXCLUDED from all training:**
| Dataset | Why excluded |
|---------|-------------|
| CREMA-D | Acted speech (91 actors performing emotions). 45.2% accuracy — the model can't distinguish acted anger from real speech because acted anger ≠ real escalation. Damages both classes. |
| SAVEE | 4 British male actors. 35.7% accuracy. Too small, too acted, too narrow. |
| TESS | 2 speakers. 98.8% accuracy = speaker memorization, not threat detection. Flatters metrics, teaches nothing. |
| RAVDESS | 24 actors. 52.1% accuracy. Same problem as CREMA-D — acted emotion ≠ real threat. |
| AudioSet: Siren | Contextual confound — siren means emergency services responding, could indicate either safe (help arriving) or unsafe (incident nearby). Neither class. |
| AudioSet: Crying/sobbing | Emotional distress ≠ physical threat. A crying person on the metro is not an unsafe situation. Conflating distress with danger weakens the unsafe class. |
| AudioSet: Fighting | AudioSet topic label, not a strongly-labeled sound event. Likely <20 usable clips in strongly-labeled CSVs. Violence dataset already covers this. |

**Critical bugs being fixed:**
1. **Data leakage**: Random per-sample split → same speaker/source in train+test. Fix: split by source/fold.
2. **Augmentation at prep time**: SpecAugment baked into .pt files once. Fix: save clean spectrograms, augment at training time.
3. **Double augmentation**: data_pipeline.py augments, then train.py augments again. Fix: single augmentation path in training loop.
4. **mix_audio/reverb at prep time**: Environmental mixing applied during prep = frozen augmentation. Fix: move to training time.

**Splitting strategy (deterministic, no leakage):**
- UrbanSound8K: use predefined folds (1-7 train, 8 val, 9-10 test)
- ESC-50: use predefined folds (1-3 train, 4 val, 5 test)
- AudioSet: `int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % 100` → 0-69 train, 70-84 val, 85-99 test
- YouTube clips: same sha256 hash on source filename → same split formula
- Violence dataset: same sha256 hash on source filename → same split formula
- IMPORTANT: Python's `hash()` is NOT deterministic across sessions (PYTHONHASHSEED). Always use `hashlib.sha256`.

---

## Current State (as of April 2026)

- 22,948 prepared .pt files (11,865/2,537/2,559 safe train/val/test, 4,172/902/913 unsafe)
- Sources: cremad(6171), yt(4712), hns(3429), bg(3000), viol(2012), tess(1600), rav(864), esc(800), savee(360)
- Removing acted speech (cremad+tess+rav+savee) = -8,995 samples (39% of data)
- Must be replaced by AudioSet downloads + FSD50K fallback

---

---

## PROMPT 1: Audit and update CLAUDE.md

```
I'm rebuilding the SafeCommute AI data pipeline from scratch. Before changing any code, I need a full audit and documentation update.

STEP 1 — Read every file (do not skip any):
- safecommute/model.py, constants.py, features.py, dataset.py, utils.py, export.py, distill.py, domain_adversarial.py
- safecommute/pipeline/download_datasets.py, data_pipeline.py, prepare_youtube_data.py, prepare_violence_data.py, validate_youtube_data.py, train.py, analyze.py, inference.py
- safecommute/benchmark/ (all files)
- CLAUDE.md, README.md, requirements.txt
- research/experiment_log.md

STEP 2 — After reading everything, rewrite CLAUDE.md with the content below. Replace the entire file:

---
# CLAUDE.md

## Project

SafeCommute AI — edge-only binary audio classifier (safe vs unsafe/escalation) for public spaces. Privacy-preserving: only non-reconstructible mel spectrograms, no raw audio storage.

Deployable to any acoustic environment via fine-tuning. Base model trained on universal threat sounds + hard negatives. Per-deployment fine-tuning adapts the "safe" class to the target environment (metro, bar, bus) in minutes.

## Architecture (DO NOT CHANGE)

- SafeCommuteCNN: CNN6 + SE + GRU + multi-scale pooling (1.83M params)
- Input: (B, 1, 64, 188) log-mel spectrogram, 3-second window @ 16kHz
- ~7ms CPU inference, 7MB float32, 5MB INT8

## Data Strategy (v2 — April 2026)

Training data is organized in three layers:

**Layer 1 — Universal threat sounds (unsafe, label=1):**
AudioSet (Screaming, Shout, Yell, Gunshot, Explosion, Glass breaking) + YouTube real screams + violence dataset

**Layer 2 — Universal hard negatives (safe, label=0):**
AudioSet (Laughter, Crowd, Speech, Music, Applause, Cheering, Singing) + ESC-50 + UrbanSound8K

**Layer 3 — Deployment-specific ambient (safe, label=0, fine-tuning only):**
Recorded in-situ audio per deployment vertical (metro rides, bar ambience, bus rides, etc.)

**DROPPED datasets:** CREMA-D, SAVEE, TESS, RAVDESS (acted speech emotions ≠ real threat audio — 35-52% accuracy proved they damage the model)

**EXCLUDED AudioSet categories:** Siren (contextual confound), Crying/sobbing (distress ≠ threat), Fighting (topic label, <20 clips)

## Critical Rules

- Split by SOURCE (predefined folds for ESC-50/UrbanSound8K, sha256 hash for everything else). Never random per-sample.
- Augmentation at TRAINING TIME only. Saved .pt files are clean un-augmented spectrograms.
- Environmental mixing (mix_audio, reverb) also at training time, not prep time.
- Use hashlib.sha256 for deterministic splits, never Python's hash().
- All scripts require PYTHONPATH=. from repo root.
- Never modify model.py architecture without explicit instruction.
- research/ is sandboxed — never modify safecommute/ from research scripts.

## Pipeline (v2)

```bash
PYTHONPATH=. python safecommute/pipeline/download_datasets.py          # ESC-50 only
PYTHONPATH=. python safecommute/pipeline/download_audioset.py          # AudioSet threat + safe
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py              # Prepare all features (clean, no augmentation)
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py       # YouTube screams + metro
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py      # Violence dataset
PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py            # Check splits, leakage, counts
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0
PYTHONPATH=. python safecommute/pipeline/analyze.py                    # Full analysis + per-source breakdown
PYTHONPATH=. python safecommute/pipeline/finetune.py --environment metro  # Fine-tune for deployment
```

## Current Model Performance

See research/experiment_log.md for full history.

## Setup

```bash
sudo pacman -S portaudio ffmpeg python-pip
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt
```
---

STEP 3 — Do NOT change any code yet. Only CLAUDE.md. Tell me exactly what you changed vs the old version.
```

---

## PROMPT 2: Clean up download_datasets.py and constants.py

```
Read safecommute/pipeline/download_datasets.py and safecommute/constants.py.

Rewrite download_datasets.py to ONLY handle ESC-50. Remove everything related to acted speech datasets.

DELETE these functions entirely:
- download_ravdess()
- download_cremad()
- download_tess()
- download_savee()
- Any helper functions only used by these (label mappers, directory creation for these datasets)

KEEP:
- download_esc50() exactly as-is (it works)
- The same script structure (progress printing, error handling, summary)

The script's main() should only call download_esc50() and print a summary.

Also check constants.py — remove any paths or references to CREMA-D, SAVEE, TESS, RAVDESS if they exist. (Currently constants.py is clean, but verify.)

After editing, show me the complete download_datasets.py so I can verify it before we continue.
```

---

## PROMPT 3: Create AudioSet download script

```
Read safecommute/constants.py, safecommute/features.py, safecommute/utils.py, and safecommute/pipeline/prepare_youtube_data.py (for style reference).

Create a new file: safecommute/pipeline/download_audioset.py

This script downloads AudioSet strongly-labeled audio clips for specific categories via yt-dlp.

THREAT CATEGORIES (saved to raw_data/audioset/threat/{category_name}/):
  "screaming":       "/m/03qc9zr"
  "shout":           "/m/07sr1lc"
  "yell":            "/m/07r660_"
  "gunshot":         "/m/032s66"
  "explosion":       "/m/014zdl"
  "glass_breaking":  "/m/07q0yl5"

HARD NEGATIVE CATEGORIES (saved to raw_data/audioset/safe/{category_name}/):
  "laughter":        "/m/01j3sz"
  "crowd":           "/m/03qtwd"
  "speech":          "/m/09x0r"
  "music":           "/m/04rlf"
  "applause":        "/m/0fx80y"
  "cheering":        "/m/03w41f"
  "singing":         "/m/015lz1"

IMPLEMENTATION DETAILS:

1. Download the AudioSet CSV metadata files (these are small text files, NOT audio):
   - http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
   - http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
   - http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
   - http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
   
   Save to raw_data/audioset/metadata/. Skip if already downloaded.
   
   WARNING: unbalanced_train_segments.csv is ~100MB. Stream-parse it line by line, do NOT load fully into memory.

2. Parse CSVs to find YouTube video IDs + start/end times for our target category IDs.
   AudioSet CSV format (after 3 comment lines): YTID, start_seconds, end_seconds, positive_labels
   Labels are quoted and comma-separated within the quotes field.

3. Download audio segments using yt-dlp:
   - Command template: yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" --download-sections "*{start}-{end}" -o "{output_path}" "https://www.youtube.com/watch?v={video_id}"
   - Save as: raw_data/audioset/{threat|safe}/{category_name}/{video_id}_{start}_{end}.wav
   - Skip if output file already exists (resume-friendly)
   - Sleep 1.5 seconds between downloads to avoid rate limiting
   - Handle failures gracefully: log to raw_data/audioset/failures.log, continue
   - Track per-category success/failure counts

4. After all downloads, print summary table:
   Category | Target | Downloaded | Failed | Skipped (existed)

CLI ARGUMENTS:
  --max-per-category INT  (default: 300) — max clips to attempt per category
  --categories LIST       (default: all) — comma-separated category names to download
  --dry-run               — only print what would be downloaded, don't download
  --sleep FLOAT           (default: 1.5) — seconds between downloads
  --threat-only           — only download threat categories
  --safe-only             — only download safe/hard-negative categories

REQUIREMENTS:
- Add yt-dlp to requirements.txt if not already there
- Import SAMPLE_RATE from safecommute.constants
- Use same code style as other pipeline scripts (sys.path.insert, seed_everything pattern)
- Add a clear docstring at the top explaining what AudioSet is and why we use strongly-labeled

After writing the file, also create a small fallback: if AudioSet yields <1000 total threat clips, we need FSD50K from Zenodo.

Create a second file: safecommute/pipeline/download_fsd50k.py
This downloads FSD50K from Zenodo (https://zenodo.org/record/4060432) as a direct ZIP download (no YouTube dependency).
FSD50K has these relevant labels for our threat class: Screaming, Shatter, Gunshot_and_gunfire, Explosion
And for safe class: Laughter, Crowd, Speech, Music, Applause
The script should:
  - Download the dev and eval sets from Zenodo
  - Parse the ground_truth CSV files to find clips matching our categories
  - Copy matching clips to raw_data/fsd50k/{threat|safe}/{category_name}/
  - Print summary

Show me both complete files.
```

---

## PROMPT 4: Rewrite data_pipeline.py — fix leakage, fix augmentation, add AudioSet

```
Read the current safecommute/pipeline/data_pipeline.py, safecommute/features.py, safecommute/dataset.py, and safecommute/constants.py carefully — all of them, completely.

This is the most critical rewrite. The current data_pipeline.py has FOUR bugs that must be fixed:

BUG 1 — DATA LEAKAGE: random_split() assigns each sample independently. Same speaker/source can be in train+test.
BUG 2 — AUGMENTATION AT PREP TIME: extract_features(y, augment=True) freezes one random augmentation per sample forever.
BUG 3 — DOUBLE AUGMENTATION: data_pipeline.py augments, then train.py augments again.
BUG 4 — MIX/REVERB AT PREP TIME: mix_audio() and add_rir_reverb() applied during prep = frozen environmental augmentation.

REWRITE data_pipeline.py with these changes:

1. REMOVE all CREMA-D, TESS, RAVDESS, SAVEE processing. Remove all label helpers (tess_label, cremad_label, SAVEE parsing, RAVDESS emotion mapping). Remove all download functions for these.

2. REMOVE mix_audio(), add_reverb_simple(), add_rir_reverb() from this script. These will move to the training loop later. The data pipeline must produce CLEAN spectrograms only.

3. KEEP and fix:
   - UrbanSound8K processing (safe class) — but split by PREDEFINED FOLDS:
     * Folds 1-7 → train, Fold 8 → val, Folds 9-10 → test
     * UrbanSound8K metadata CSV has a "fold" column — use it
     * Keep SAFE_BG_LABELS and HARD_NEG_LABELS for filtering
   - ESC-50 processing (safe class) — split by PREDEFINED FOLDS:
     * Folds 1-3 → train, Fold 4 → val, Fold 5 → test
     * ESC-50 metadata has a "fold" column
     * Keep ESC_SAFE_AMBIENT and ESC_HARD_NEG for filtering

4. ADD AudioSet processing (new):
   - Look for raw_data/audioset/threat/{category}/*.wav → label=1 (unsafe)
   - Look for raw_data/audioset/safe/{category}/*.wav → label=0 (safe)
   - For each wav: load at 16kHz, pad_or_truncate, extract_features(y, augment=False)
   - Split using sha256 hash: int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
     * 0-69 → train, 70-84 → val, 85-99 → test

5. ADD FSD50K processing (if directory exists):
   - Same structure as AudioSet: raw_data/fsd50k/threat/...  and raw_data/fsd50k/safe/...
   - Same sha256-based splitting

6. FIX extract_features() call — ALWAYS pass augment=False:
   features = extract_features(y, augment=False)
   No exceptions. Every saved .pt file is a clean spectrogram.

7. ALSO update safecommute/features.py:
   - Add a prominent docstring to extract_features(): 
     "WARNING: augment=True should ONLY be used at training time in the DataLoader, NEVER during data preparation. All .pt files must be clean spectrograms."
   - Keep the augment parameter for backward compatibility (it's used in the training loop)

8. Keep compute_and_save_stats() — this computes dataset-wide mean/std for normalization.

9. At the end of the script, print a summary table:
   Source | Safe(train/val/test) | Unsafe(train/val/test) | Split method

Show me the complete data_pipeline.py and features.py when done.
```

---

## PROMPT 5: Fix prepare_youtube_data.py and prepare_violence_data.py

```
Read safecommute/pipeline/prepare_youtube_data.py and safecommute/pipeline/prepare_violence_data.py completely.

Both files have the same two bugs:

BUG 1 — SPLIT BY CHUNK, NOT SOURCE: Random split per chunk. Different chunks from the SAME source file can end up in train and test = data leakage. Model sees part of an audio recording in training, tested on another part of the same recording.

BUG 2 — AUGMENTATION AT PREP TIME: extract_features(chunk, augment=(split == 'train')). Same baked-augmentation bug.

Fix BOTH files:

FOR prepare_youtube_data.py:
- Split by SOURCE FILE (the original .wav), not by chunk
- Use deterministic sha256 hash: int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
  * 0-69 → train, 70-84 → val, 85-99 → test
- ALL chunks from the same source file go to the SAME split
- Call extract_features(chunk, augment=False) ALWAYS
- Keep everything else: chunk_audio(), process_directory(), metro/scream handling

FOR prepare_violence_data.py:
- Split by SOURCE FILE (violence_IDX_LABEL.wav), not by chunk
- Same sha256-based split formula
- ALL chunks from the same source file go to the SAME split
- Call extract_features(chunk, augment=False) ALWAYS
- Keep everything else

IMPORTANT: Add `import hashlib` at the top of both files.

Show me both complete files when done.
```

---

## PROMPT 6: Fix train.py and dataset.py — move all augmentation to training time

```
Read safecommute/pipeline/train.py and safecommute/dataset.py completely.

Now that data prep saves clean spectrograms, ALL augmentation must happen during training. The current code has two problems:
1. The per-sample CPU loop in train.py (lines 134-139) is extremely slow
2. dataset.py has no augmentation capability

Make these changes:

IN dataset.py — add training-time augmentation:
- Add an `augment` parameter to TensorAudioDataset.__init__ (default False)
- Add an `apply_env_aug` parameter (default False) — for environmental mixing at training time
- When augment=True, apply in __getitem__:
  * FrequencyMasking(freq_mask_param=10) with 50% probability
  * TimeMasking(time_mask_param=20) with 50% probability
  These match what features.py previously did, but now it's different every epoch.
- Do NOT add environmental mixing to __getitem__ — that's too slow per-sample

IN train.py — fix the augmentation flow:
- Create train_dataset with augment=True: TensorAudioDataset(train_dir, mean, std, augment=True)
- Val and test datasets: augment=False (default)
- DELETE the per-sample CPU loop entirely:
  ```python
  # DELETE ALL OF THIS:
  if use_strong_aug:
      aug_inputs = []
      for i in range(inputs.size(0)):
          aug_inputs.append(spec_augment_strong(inputs[i].cpu()).to(device))
      inputs = torch.stack(aug_inputs)
  ```
- REPLACE with batch-level GPU augmentation (fast, no CPU transfer):
  ```python
  if use_strong_aug:
      # Additional strong augmentation on GPU (on top of dataset-level augmentation)
      if random.random() < 0.3:
          inputs = inputs + torch.randn_like(inputs) * 0.1  # Gaussian noise
      if random.random() < 0.3:
          shift = random.randint(-20, 20)
          inputs = torch.roll(inputs, shifts=shift, dims=-1)  # Circular time shift
      if random.random() < 0.2:
          # Random frequency band dropout (stronger than single mask)
          f_start = random.randint(0, 50)
          f_width = random.randint(3, 10)
          inputs[:, :, f_start:f_start+f_width, :] = 0
  ```
  This is ~10x faster than the per-sample CPU loop.

- Keep everything else: FocalLoss, mixup, cosine annealing, early stopping, evaluation, argparse

Show me both complete files (dataset.py and train.py).
```

---

## PROMPT 7: Create verification script and consistency check

```
Read ALL modified files to verify consistency:
- safecommute/pipeline/download_audioset.py
- safecommute/pipeline/download_fsd50k.py
- safecommute/pipeline/download_datasets.py
- safecommute/pipeline/data_pipeline.py
- safecommute/pipeline/prepare_youtube_data.py
- safecommute/pipeline/prepare_violence_data.py
- safecommute/pipeline/train.py
- safecommute/dataset.py
- safecommute/features.py
- safecommute/constants.py
- requirements.txt

CHECK for these issues (fix any you find):
1. Import errors — no references to deleted functions (download_ravdess, download_cremad, tess_label, cremad_label, etc.)
2. Path consistency — download_audioset.py saves to raw_data/audioset/{threat|safe}/{category}/ and data_pipeline.py reads from that exact path
3. Splitting formula — every script uses the SAME sha256 formula: int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100 with 0-69/70-84/85-99
4. Augmentation — NO script except train.py (via dataset.py) calls extract_features with augment=True
5. requirements.txt has yt-dlp added
6. No references to CREMA-D, SAVEE, TESS, RAVDESS in any pipeline script

After fixing issues, create: safecommute/pipeline/verify_pipeline.py

This diagnostic script checks:
1. Raw data directories exist and contain expected files:
   - raw_data/audioset/threat/screaming/*.wav (and other categories)
   - raw_data/audioset/safe/laughter/*.wav (and other categories)
   - raw_data/youtube_screams/*.wav
   - raw_data/youtube_metro/*.wav
   - raw_data/violence/*.wav
   - Print count per directory

2. Prepared data has correct structure:
   - prepared_data/{train,val,test}/{0_safe,1_unsafe}/*.pt
   - Count per split per class

3. Source-level leakage check:
   - Extract source identifiers from .pt filenames
   - Verify no source appears in multiple splits
   - Print any violations found

4. Class balance report:
   - Per-split class distribution
   - Flag if any split is >85% one class (indicates balance problem)

5. Summary table with all counts

This is diagnostic only — it does not modify any data. Show me the complete file.
```

---

## PROMPT 8: Run the data pipeline end to end

```
Time to run everything. Execute in order, checking output after each step:

STEP 1 — Check current raw data:
  ls -la raw_data/
  ls -la raw_data/audioset/ 2>/dev/null
  find raw_data/youtube_screams -name "*.wav" | wc -l
  find raw_data/youtube_metro -name "*.wav" | wc -l
  find raw_data/violence -name "*.wav" | wc -l

STEP 2 — Download ESC-50 (fast, direct download):
  PYTHONPATH=. python safecommute/pipeline/download_datasets.py

STEP 3 — Download AudioSet clips (this takes 1-3 hours):
  PYTHONPATH=. python safecommute/pipeline/download_audioset.py --max-per-category 300 --threat-only
  
  Check output. If >60% failure rate, try:
  PYTHONPATH=. python safecommute/pipeline/download_audioset.py --max-per-category 300 --threat-only --sleep 3
  
  Then safe categories:
  PYTHONPATH=. python safecommute/pipeline/download_audioset.py --max-per-category 300 --safe-only

  If total AudioSet threat clips < 500, run FSD50K fallback:
  PYTHONPATH=. python safecommute/pipeline/download_fsd50k.py

STEP 4 — Clear old prepared data (it has leakage + baked augmentation):
  rm -rf prepared_data/
  
  (Confirm with me before running this — I want to verify raw_data has enough new data first)

STEP 5 — Run the data pipeline:
  PYTHONPATH=. python safecommute/pipeline/data_pipeline.py

STEP 6 — Process YouTube and violence data:
  PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py
  PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py

STEP 7 — Verify:
  PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py

Report the final verification output in full. I need to see:
- Total samples per split per class
- Per-source breakdown
- Leakage check result (must be ZERO violations)
- Class balance

CRITICAL: If total training samples < 5,000, STOP and report. We need to either download more AudioSet data (increase --max-per-category) or add FSD50K. Do not proceed to training with insufficient data.
```

---

## PROMPT 9: Train, evaluate, and compare to old results

```
Read safecommute/pipeline/train.py and safecommute/pipeline/analyze.py.

STEP 1 — Report dataset size before training:
  find prepared_data/train/0_safe -name "*.pt" | wc -l
  find prepared_data/train/1_unsafe -name "*.pt" | wc -l
  find prepared_data/val -name "*.pt" | wc -l
  find prepared_data/test -name "*.pt" | wc -l

STEP 2 — Train the model (3 seeds for confidence intervals):
  PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0 --save models/v2_seed42.pth
  PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0 --save models/v2_seed123.pth
  PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0 --save models/v2_seed7.pth
  
  NOTE: train.py currently uses seed_everything() with seed=42. For multiple seeds, you'll need to:
  - Add a --seed argument to train.py if it doesn't have one
  - Pass --seed 42, --seed 123, --seed 7

  For each run, report: final test AUC, accuracy, F1.
  Then report: mean ± std across all 3 seeds.

STEP 3 — Copy best model (highest val AUC) to safecommute_edge_model.pth for analysis:
  cp models/v2_seed{BEST}.pth safecommute_edge_model.pth

STEP 4 — Run full analysis:
  PYTHONPATH=. python safecommute/pipeline/analyze.py

  Report in full:
  - Train/Val/Test: AUC, Accuracy, F1, Precision, Recall
  - Train-test AUC gap (should be < 0.05)
  - Per-source accuracy breakdown
  - Optimal threshold from ROC analysis

STEP 5 — Compare against v1 results:
  Old model (v1, with acted speech):
  - AUC=0.950, Accuracy=0.745, F1=0.760
  - CREMA-D: 45.2%, SAVEE: 35.7%, TESS: 98.8%, RAVDESS: 52.1%
  - YouTube screams: 97.3%, metro: 90.3%
  - LOSO mean AUC: 0.767

  The new model should:
  - Have AUC ≥ 0.90 (if lower, something is wrong — check data)
  - Have NO source below 60% accuracy (we removed the worst offenders)
  - Show a smaller train-test gap (no leakage)
  - Per-source accuracy should be more uniform

STEP 6 — If results are bad (AUC < 0.90):
  Check these in order:
  1. Is training data < 5000 samples? → need more AudioSet/FSD50K
  2. Is class balance > 85:15? → need more unsafe samples
  3. Is train accuracy much higher than val? → augmentation may be too weak
  4. Try gamma 2.0 instead of 3.0

STEP 7 — Update research/experiment_log.md:
  Add new section "## v2 Pipeline: AudioSet-based, No Acted Speech"
  Record all metrics in the same table format as existing entries.
  Note: "Dropped CREMA-D/SAVEE/TESS/RAVDESS. Added AudioSet. Fixed data leakage (source-aware splits). Fixed augmentation timing (training-time only). 3-seed evaluation."
```

---

## PROMPT 10: LOSO evaluation on new data

```
Read research/experiments/loso_evaluation.py to understand the existing LOSO infrastructure.

Run Leave-One-Source-Out (LOSO) evaluation on the v2 dataset. This is ESSENTIAL for the paper — it answers: "Does this model generalize to audio sources it has never seen during training?"

If the existing loso_evaluation.py works with the new data format, use it. If not, fix it to work with v2. The script should:

1. Identify all unique sources in the prepared data (from .pt filename prefixes)
2. For each source S:
   a. Train on all data EXCEPT source S
   b. Evaluate on source S only
   c. Record AUC, accuracy, F1

3. Report:
   - Per-source LOSO results table
   - Mean LOSO AUC across all sources
   - Which source is hardest to generalize to (lowest AUC when held out)
   - Which source is easiest (highest AUC when held out)

4. Compare to v1 LOSO (mean AUC=0.767, YouTube generalized best at 0.803):
   - Did removing acted speech improve cross-source generalization?
   - Is the new LOSO AUC higher than 0.767?

Run it:
  PYTHONPATH=. python research/experiments/loso_evaluation.py

Report all results. Update research/experiment_log.md with LOSO v2 results.

IMPORTANT: LOSO training is expensive (one full training run per source). If there are >8 sources, this could take hours. That's expected — do not skip sources.
```

---

## PROMPT 11: Ablation study and SOTA benchmark

```
Read research/experiments/ablation_study.py and safecommute/benchmark/run_benchmark.py.

PART A — ABLATION STUDY

Train 4 model variants to prove each component matters. Each variant removes one component:

1. Full model (baseline) — normal SafeCommuteCNN training
2. No SE attention — skip SE blocks (replace with identity)
3. No GRU — replace GRU + multi-scale pooling with global average pooling on CNN output
4. No multi-scale pooling — use only GRU last hidden state (128-dim) instead of concat(last, mean, max) (384-dim)

Rules:
- Do NOT modify safecommute/model.py. Create variant models in the ablation script by subclassing or monkey-patching.
- Same hyperparameters for all: --focal --cosine --strong-aug --gamma 3.0
- Same seed (42) for fair comparison
- Report for each: AUC, Accuracy, F1, Params count, delta-AUC vs full model

If the existing ablation_study.py works with v2 data, use it. If not, fix it.

Run: PYTHONPATH=. python research/experiments/ablation_study.py

PART B — SOTA COMPARISON

Run pretrained PANNs (CNN14, 80M params, pretrained on full AudioSet) on our exact test set. This proves: "Our 1.8M param model is competitive with an 80M param model on this specific task."

Read safecommute/benchmark/ to see existing infrastructure. Run:
  PYTHONPATH=. python safecommute/benchmark/run_benchmark.py

If this doesn't work with v2 data paths, fix it.

The comparison table for the paper should be:
| Model | Params | Size | Latency | AUC | Accuracy | F1 |
|-------|--------|------|---------|-----|----------|-----|
| PANNs CNN14 | 80M | 320MB | ~200ms | ? | ? | ? |
| SafeCommuteCNN | 1.83M | 7MB | ~7ms | ? | ? | ? |
| Energy baseline | 0 | 0 | <1ms | ? | ? | ? |

Update research/experiment_log.md with both ablation and SOTA results.
```++++++

---

## PROMPT 12: Create fine-tuning script for deployment personalization

```
Read safecommute/pipeline/train.py, safecommute/dataset.py, and safecommute/constants.py.

Create: safecommute/pipeline/finetune.py

This script fine-tunes the base model for a specific deployment environment. It takes recorded ambient audio and adapts the model's "safe" class to that environment.

USAGE:
  # Fine-tune for metro deployment using recorded metro audio
  PYTHONPATH=. python safecommute/pipeline/finetune.py \
    --base-model safecommute_edge_model.pth \
    --environment metro \
    --ambient-dir raw_data/youtube_metro \
    --epochs 10 \
    --lr 1e-4 \
    --freeze-cnn

IMPLEMENTATION:

1. Load the base model (trained on Layer 1+2)
2. Load ambient audio from --ambient-dir (all files treated as safe, label=0)
3. Process ambient audio: load → pad_or_truncate → extract_features(augment=False) → save to a temp directory
4. Create a fine-tuning dataset:
   - Safe class: the new ambient audio + a random subset (50%) of the base training safe data (to prevent catastrophic forgetting)
   - Unsafe class: ALL base training unsafe data (threat sounds don't change per deployment)
5. Fine-tune:
   - If --freeze-cnn: freeze all Conv2d and BatchNorm2d parameters. Only train GRU + FC layers.
   - If not: train everything with low LR
   - Use same FocalLoss + cosine schedule as base training
   - Default: 10 epochs (fine-tuning converges fast)
6. Evaluate on the base test set (threat detection must not degrade)
7. Save as models/{environment}_model.pth
8. Print comparison: base model AUC vs fine-tuned AUC on test set
9. Clean up temp directory

ARGUMENTS:
  --base-model PATH        Base model checkpoint (default: safecommute_edge_model.pth)
  --environment NAME       Environment name (metro, bar, bus, etc.) — used for output naming
  --ambient-dir PATH       Directory with ambient .wav files for this environment
  --epochs INT             Fine-tuning epochs (default: 10)
  --lr FLOAT               Learning rate (default: 1e-4, lower than base training)
  --freeze-cnn             Freeze CNN layers, only train GRU+FC (faster, less forgetting)
  --keep-safe-ratio FLOAT  Ratio of base safe data to keep (default: 0.5)

Show me the complete file.
```

---

## PROMPT 13: Create metro deployment test script

```
Read safecommute/pipeline/inference.py, safecommute/export.py, safecommute/features.py, and safecommute/constants.py.

Create: safecommute/pipeline/test_deployment.py

This is the acceptance test script. It validates that the model is deployment-ready for metro by running a comprehensive battery of tests on actual audio. It's what you run before shipping.

TESTS TO IMPLEMENT:

TEST 1 — THREAT DETECTION (must-pass):
  Load each .wav from a test directory (default: raw_data/youtube_screams/)
  Run inference on 3-second windows with 1-second stride
  For each file: compute max probability, mean probability, % of windows above threshold
  PASS criteria: ≥ 90% of threat files have at least one window above threshold (0.5 default)

TEST 2 — FALSE POSITIVE RATE ON AMBIENT (must-pass):
  Load each .wav from ambient directory (default: raw_data/youtube_metro/)
  Same sliding window inference
  PASS criteria: ≤ 5% of ambient files trigger a false positive (any window above threshold)

TEST 3 — LATENCY (must-pass):
  Run 1000 inference passes on random input tensors
  Measure mean and p99 latency
  PASS criteria: mean < 15ms, p99 < 30ms on CPU

TEST 4 — MODEL SIZE (must-pass):
  Check model file sizes (float32, INT8 if available, ONNX if available)
  PASS criteria: float32 ≤ 10MB, INT8 ≤ 6MB

TEST 5 — CONSISTENCY (must-pass):
  Run the same audio through the model 10 times
  Verify outputs are identical (deterministic inference)
  PASS criteria: max absolute difference across runs = 0.0

TEST 6 — SILENCE HANDLING (must-pass):
  Feed 3 seconds of silence (all zeros)
  Feed 3 seconds of very quiet noise (RMS < 0.001)
  PASS criteria: both should predict safe with probability > 0.8
  (The model should not hallucinate threats in silence)

TEST 7 — EXPORT VERIFICATION (if export files exist):
  Load ONNX model if it exists, run same test inputs
  Compare ONNX output to PyTorch output
  PASS criteria: max absolute difference < 0.01

OUTPUT:
  Print a clear test report:
  ✓ PASS  Threat detection: 95.6% detection rate (target ≥ 90%)
  ✓ PASS  False positive rate: 3.2% (target ≤ 5%)
  ✓ PASS  Latency: mean=6.8ms, p99=12.1ms (target mean<15ms, p99<30ms)
  ✓ PASS  Model size: 6.98MB float32, 4.91MB INT8 (target ≤10MB, ≤6MB)
  ✓ PASS  Consistency: max diff = 0.0
  ✓ PASS  Silence handling: safe prob = 0.94 / 0.91
  ✗ FAIL  ONNX verification: max diff = 0.03 (target < 0.01)
  
  RESULT: 6/7 passed, 1/7 failed

  Exit code 0 if all must-pass tests pass, exit code 1 otherwise.

ARGUMENTS:
  --model PATH             Model to test (default: safecommute_edge_model.pth)
  --threat-dir PATH        Directory with threat audio (default: raw_data/youtube_screams/)
  --ambient-dir PATH       Directory with ambient audio (default: raw_data/youtube_metro/)
  --threshold FLOAT        Detection threshold (default: 0.5)
  --verbose                Print per-file results

Show me the complete file.
```

---

## PROMPT 14: Generate paper-ready figures, update all documentation, final export

```
Read safecommute/pipeline/analyze.py, safecommute/export.py, and research/experiment_log.md.

PART A — Paper figures

Create: research/generate_paper_figures.py

This script generates publication-quality figures. Use matplotlib with:
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.size'] = 11
  Clean, minimal academic style. No emojis. No casual formatting.

Generate these figures (save to research/figures/ at 300 DPI):

1. per_source_accuracy.png — Horizontal bar chart of per-source accuracy. Sorted by accuracy. Color-coded: environmental=steelblue, real-world=forestgreen, AudioSet=darkorange. Sample count per source shown on bars.

2. roc_curves.png — Train/val/test ROC curves overlaid. AUC values in legend. Clean axes.

3. confusion_matrix.png — Test set only. Counts + percentages. Clean 2x2 grid.

4. data_distribution.png — Training data composition by source. Horizontal stacked bar or treemap.

5. ablation_table.tex — LaTeX table from ablation results. Columns: Model Variant, Params, AUC, Acc, F1, ΔAUC.

6. efficiency_table.tex — LaTeX comparison table: our model vs PANNs vs Energy baseline.

7. loso_results.png — Bar chart of LOSO AUC per held-out source. Horizontal line at mean. Error bars if available.

8. adaptation_curve.png — (if finetune results exist) Plot AUC vs number of fine-tuning epochs for metro adaptation.

Run: PYTHONPATH=. python research/generate_paper_figures.py

PART B — Export final model

  PYTHONPATH=. python -m safecommute.export

  Report: float32 size, INT8 size, ONNX size, TorchScript size, latency (mean, p99)

PART C — Run deployment test

  PYTHONPATH=. python safecommute/pipeline/test_deployment.py --verbose

  Report full test results.

PART D — Update documentation

Update README.md:
  - Remove all CREMA-D, SAVEE, TESS, RAVDESS references
  - Update results table with v2 numbers (mean ± std from 3 seeds)
  - Update per-source accuracy (only real sources now)
  - Update data section with Layer 1/2/3 explanation
  - Add fine-tuning section explaining personalization
  - Update quick start commands (add download_audioset.py)
  - Keep architecture section unchanged

Update research/experiment_log.md:
  - Ensure all v2 experiments are recorded
  - Add final summary section

Create research/data_sources.md:
  For each dataset used (AudioSet, ESC-50, UrbanSound8K, YouTube clips, violence dataset, FSD50K if used):
  - Full academic citation (author, title, year, venue/journal)
  - License
  - URL
  - Number of samples used
  - Label mapping (which categories → safe/unsafe)
  - Any filtering or quality control applied
  
  This is required for the paper's data section.

PART E — Print final paper summary

Print a complete summary of everything needed for the paper:
  MODEL:
  - Architecture: CNN6 + SE + GRU + multi-scale pooling
  - Parameters: 1.83M
  - Size: Xmb float32, Xmb INT8, Xmb ONNX
  - Latency: Xms mean, Xms p99

  DATA:
  - Training: X samples (Y safe, Z unsafe) from N sources
  - Validation: X samples
  - Test: X samples
  - Sources: [list with counts]

  RESULTS (mean ± std, 3 seeds):
  - AUC: X ± Y
  - Accuracy: X ± Y
  - F1: X ± Y
  - Train-test gap: X

  LOSO: mean AUC = X (range: Y - Z)
  
  ABLATION: [table]
  
  SOTA: [comparison table]
  
  DEPLOYMENT:
  - Threat detection rate: X%
  - False positive rate: X%
  - All deployment tests: PASS/FAIL
```

---

## Execution Notes

### Order
Run prompts 1→14 sequentially. Do not skip. Each builds on the previous.

### If AudioSet downloads fail badly (>70% failure rate)
This is the biggest risk. YouTube videos get deleted. Mitigations:
1. FSD50K fallback is built into Prompt 3 — use it
2. Lower --max-per-category and increase --sleep
3. If total threat clips < 500, the model WILL underperform — you need real threat audio

### If model performance drops below AUC 0.90 after retraining
1. Check total training samples (need >5,000)
2. Check class balance (should be 55-75% safe)
3. Try gamma 2.0 instead of 3.0
4. Check augmentation is working: train accuracy should be LOWER than val accuracy in early epochs if augmentation is strong
5. If still bad: temporarily add CREMA-D angry+fear samples at 0.3x loss weight as supplement

### Recording your own metro audio (not covered in prompts — physical task)
- Record 3-4 hours of metro rides with phone in pocket (realistic placement)
- Record at stations (announcements, crowd), in trains (motion, doors), and platforms
- Save as WAV 16kHz mono
- Put in raw_data/recorded_metro/
- Use finetune.py to adapt the base model

### Paper structure (for reference when writing)
1. Introduction: audio safety monitoring, privacy constraints, edge deployment
2. Related work: PANNs, AudioSet, sound event detection, domain adaptation
3. Method: architecture, training strategy (focal loss, augmentation, mixup), personalization via fine-tuning
4. Data: Layer 1/2/3 strategy, source-aware splitting, why acted speech was excluded
5. Results: main metrics, per-source, LOSO, ablation, SOTA comparison
6. Deployment: latency, model size, acceptance tests, fine-tuning protocol
7. Conclusion + limitations (acted speech gap, AudioSet coverage, single-channel)
