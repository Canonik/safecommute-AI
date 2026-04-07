# PROMPT 16: Fix probability calibration — the model can rank but can't decide

## The Problem

The model has AUC=0.856 (good ranking) but its raw probabilities are **completely useless for deployment**. Diagnosis:

```
Safe samples:   mean prob = 0.511, median = 0.531 (should be ~0.1-0.2)
Unsafe samples: mean prob = 0.669, median = 0.674 (should be ~0.8-0.9)

At threshold 0.5:
  Safe correct:   42.5%  ← the model calls 57.5% of SAFE audio as UNSAFE
  Unsafe correct: 96.5%  ← good, but at the cost of massive false alarms

Silence:        prob = 0.54  ← should be ~0.0
Quiet room:     prob = 0.24  ← OK
Any real sound:  prob = 0.5-0.7  ← everything clusters here
```

The entire probability distribution is compressed into the 0.4-0.7 range. Safe and unsafe samples overlap almost completely. demo.py sits at 0.67 because that's where EVERYTHING lands — the model can't produce confident "definitely safe" or "definitely unsafe" predictions.

## Root Cause

**Focal loss with gamma=3 combined with label smoothing=0.1.**

Focal loss multiplies the cross-entropy by `(1 - p_t)^gamma`. With gamma=3:
- A sample the model predicts at 70% confidence gets weight `(1-0.7)^3 = 0.027` — nearly zero
- Only samples the model is truly confused about (50/50) get meaningful gradient
- The model learns to stay near 50/50 for everything because that's where the gradient lives
- Label smoothing further prevents sharp logits by softening targets to [0.05, 0.95]

This is a known failure mode of focal loss: at high gamma on moderately-sized datasets, it **discourages confident predictions**. The model learns good relative ordering (AUC) but terrible absolute calibration.

## The Fix

Three changes, executed in order:

### STEP 1 — Retrain with gamma=1 and no label smoothing

Read safecommute/pipeline/train.py.

Retrain the base model with these changes to the command:

```bash
PYTHONPATH=. python safecommute/pipeline/train.py \
    --focal --cosine --strong-aug --gamma 1.0 --seed 42 \
    --save safecommute_edge_model.pth
```

gamma=1.0 is much milder than 3.0 — it still helps with class imbalance (via the alpha class weights) but doesn't crush the gradient on semi-confident predictions. The model should now learn to output 0.1 for obviously safe audio and 0.9 for obviously unsafe audio.

BUT FIRST: modify train.py to accept a `--label-smoothing` argument (default 0.0, was hardcoded 0.1):
- Add `parser.add_argument('--label-smoothing', type=float, default=0.0)`
- Pass it to FocalLoss and CrossEntropyLoss instead of the hardcoded `LABEL_SMOOTHING = 0.1`
- For this retrain, use `--label-smoothing 0.0` (no smoothing — let the model be confident)

After training, run the same probability distribution diagnostic:
```python
PYTHONPATH=. python -c "
import torch, json, numpy as np, os
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import SAMPLE_RATE, MODEL_SAVE_PATH, STATS_PATH, DATA_DIR

with open(STATS_PATH) as f:
    s = json.load(f)
mean, std = s['mean'], s['std']

model = SafeCommuteCNN()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
model.eval()

ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
safe_probs, unsafe_probs = [], []
with torch.no_grad():
    for i in range(len(ds)):
        feat, label = ds[i]
        p = torch.softmax(model(feat.unsqueeze(0)), dim=1)[0][1].item()
        (unsafe_probs if label == 1 else safe_probs).append(p)

print(f'Safe (n={len(safe_probs)}):   mean={np.mean(safe_probs):.3f}  median={np.median(safe_probs):.3f}  p10={np.percentile(safe_probs,10):.3f}  p90={np.percentile(safe_probs,90):.3f}')
print(f'Unsafe (n={len(unsafe_probs)}): mean={np.mean(unsafe_probs):.3f}  median={np.median(unsafe_probs):.3f}  p10={np.percentile(unsafe_probs,10):.3f}  p90={np.percentile(unsafe_probs,90):.3f}')
print(f'Separation: {np.mean(unsafe_probs) - np.mean(safe_probs):.3f} (want > 0.4)')
print(f'Safe correct at 0.5: {100*sum(1 for p in safe_probs if p < 0.5)/len(safe_probs):.1f}%')
print(f'Unsafe correct at 0.5: {100*sum(1 for p in unsafe_probs if p > 0.5)/len(unsafe_probs):.1f}%')
"
```

**Target**: Safe mean < 0.3, unsafe mean > 0.7, separation > 0.4. If the separation is still < 0.3, try gamma=0 (pure weighted cross-entropy, no focal at all) with `--no-focal`:
```bash
PYTHONPATH=. python safecommute/pipeline/train.py \
    --cosine --strong-aug --label-smoothing 0.0 --seed 42 \
    --save safecommute_edge_model.pth
```

### STEP 2 — Apply temperature scaling (post-hoc calibration)

Even after fixing gamma, the raw probabilities may not be perfectly calibrated. Temperature scaling is a one-parameter post-hoc fix that sharpens or softens the probability distribution without changing ranking (AUC).

Create: safecommute/pipeline/calibrate.py

This script:
1. Loads the trained model
2. Runs inference on the val set to get all logits
3. Optimizes a single temperature parameter T to minimize NLL on val set:
   `calibrated_probs = softmax(logits / T)`
   If T < 1, predictions become sharper (more confident)
   If T > 1, predictions become softer (less confident)
4. Saves T to models/temperature.json
5. Reports calibration metrics (ECE before/after) and probability distributions

```python
# The optimization is simple:
# For each T in [0.1, 0.2, ..., 5.0], compute NLL on val set
# Pick the T that minimizes NLL
# Alternatively, use scipy.optimize.minimize_scalar for exact T
```

Then update preprocess-based inference (demo.py, inference.py, test_deployment.py) to apply temperature:
```python
logits = model(feat)
logits = logits / temperature  # temperature scaling
prob = torch.softmax(logits, dim=1)[0][1].item()
```

Temperature should be loaded from models/temperature.json alongside the model.

### STEP 3 — Fix demo.py to show calibrated, useful probabilities

After steps 1-2, update demo.py:
1. Load temperature from models/temperature.json if it exists
2. Apply `logits / T` before softmax
3. The thresholds should now work intuitively:
   - Silence → prob ~0.0-0.1 → SAFE
   - Normal room → prob ~0.1-0.3 → SAFE
   - Loud crowd → prob ~0.3-0.5 → SAFE
   - Raised voice → prob ~0.5-0.7 → WARNING
   - Scream/gunshot → prob ~0.8-1.0 → ALERT

### STEP 4 — Re-run everything and verify

1. Retrain: `PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 1.0 --label-smoothing 0.0 --seed 42`
2. Calibrate: `PYTHONPATH=. python safecommute/pipeline/calibrate.py`
3. Export: `PYTHONPATH=. python -m safecommute.export`
4. Fine-tune metro: `PYTHONPATH=. python safecommute/pipeline/finetune.py --environment metro --ambient-dir raw_data/youtube_metro --warmup-epochs 5 --lr 5e-5`
5. Deploy test: `PYTHONPATH=. python safecommute/pipeline/test_deployment.py --model models/metro_model.pth --thresholds-file models/metro_thresholds.json`
6. Demo test: `PYTHONPATH=. python demo.py --model models/metro_model.pth --duration 15`

Report for each step:
- Probability distribution (safe mean, unsafe mean, separation)
- AUC (should stay >= 0.83)
- Deployment test results (all 7 tests)
- Demo behavior (does it show SAFE in a quiet room?)

### SUCCESS CRITERIA

The fix is successful when:
- [ ] Safe samples: mean prob < 0.30, p90 < 0.50
- [ ] Unsafe samples: mean prob > 0.70, p10 > 0.50
- [ ] Separation (unsafe mean - safe mean) > 0.40
- [ ] Silence: prob < 0.10
- [ ] Quiet room: prob < 0.20
- [ ] demo.py shows SAFE in a quiet room, WARNING/ALERT only on actual loud sounds
- [ ] AUC >= 0.83 (ranking should not degrade)
- [ ] Deployment test: threat detection >= 85%, FP rate <= 10%

### IF STEP 1 ALONE IS SUFFICIENT

It's possible that just gamma=1 + no label smoothing fixes calibration completely, making temperature scaling unnecessary. If the probability diagnostic after Step 1 already meets the success criteria, skip Steps 2-3 and go straight to Step 4.

### WHY THIS MATTERS FOR THE PAPER

The paper needs to present a deployable system, not just AUC numbers. A reviewer will ask "what happens when you actually run this?" and demo.py needs to convincingly show Safe/Warning/Alert behavior. Poor calibration is the #1 reason ML models fail in production even with good AUC — this fix directly addresses that.

Additionally, for the paper you can present:
- Calibration comparison: gamma=3 vs gamma=1 probability distributions (a compelling figure)
- ECE (Expected Calibration Error) before/after temperature scaling
- The insight that "focal loss gamma must be tuned for calibration, not just AUC"
