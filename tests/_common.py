"""
Shared helpers for the tests/ suite.

Purpose: provide a single import surface for model loading, inference,
per-source bookkeeping, Phase A metric computation, latency timing, and
hardware disclosure — so tests/verify_performance_claims.py and any future
verification script do not reimplement the same code.

The existing scripts (analyze_model.py, validate_fp_claims.py,
measure_latency.py, measure_latency_opt.py) are left unchanged so the
Step 0 baselines continue to match byte-for-byte. New code imports from
here.

Every numeric result is hardware-dependent where relevant; pair every
latency report with hw_disclosure().
"""

import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN  # noqa: E402
from safecommute.dataset import TensorAudioDataset  # noqa: E402
from safecommute.constants import (  # noqa: E402
    DATA_DIR, MODEL_SAVE_PATH, N_MELS, STATS_PATH, TIME_FRAMES,
)


# ──────────────────────────────────────────────────────────────────────
# Model + data loading
# ──────────────────────────────────────────────────────────────────────

def load_stats() -> Tuple[float, float]:
    """Return (mean, std) from feature_stats.json, fallbacks (0, 1)."""
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return float(s['mean']), float(s['std'])


def load_model(checkpoint_path: str = MODEL_SAVE_PATH,
               device: Optional[torch.device] = None) -> SafeCommuteCNN:
    """Load a SafeCommuteCNN from a .pth checkpoint. Raises on missing file."""
    if device is None:
        device = torch.device('cpu')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_test_dataset(split: str = 'test') -> TensorAudioDataset:
    """Load prepared_data/<split>/ with the repo's normalization stats."""
    mean, std = load_stats()
    return TensorAudioDataset(
        os.path.join(DATA_DIR, split), mean, std, augment=False)


# ──────────────────────────────────────────────────────────────────────
# Source bookkeeping — the filename → source prefix rule appears in three
# places today; this is the single source of truth.
# ──────────────────────────────────────────────────────────────────────

def source_for(name: str) -> str:
    """Collapse 'as_speech_XYZ.pt' → 'as_speech', 'bg_12345.pt' → 'bg'."""
    base = os.path.basename(name)
    # Drop any .pt / .wav suffix before splitting.
    for ext in ('.pt', '.wav'):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split('_')
    if len(parts) >= 2 and parts[1] and not parts[1][0].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def safe_indices_by_prefix(dataset: TensorAudioDataset, prefix: str) -> List[int]:
    """Indices of safe-class samples whose filename starts with `prefix`."""
    out = []
    for i, (fp, label) in enumerate(zip(dataset.filepaths, dataset.labels)):
        if label != 0:
            continue
        if os.path.basename(fp).startswith(prefix):
            out.append(i)
    return out


def all_safe_indices(dataset: TensorAudioDataset) -> List[int]:
    return [i for i, l in enumerate(dataset.labels) if l == 0]


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def run_inference(model: SafeCommuteCNN,
                  dataset: TensorAudioDataset,
                  device: torch.device,
                  batch_size: int = 64,
                  num_workers: int = 2,
                  indices: Optional[List[int]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (probs_unsafe, labels, basenames) aligned by index.

    If `indices` is provided, runs only on that subset — used for per-source
    FPR computations.
    """
    ds = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch[0] if isinstance(batch, (list, tuple)) else batch
            lab = batch[1] if isinstance(batch, (list, tuple)) else None
            logits = model(inp.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            if lab is not None:
                labels.append(lab.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels) if labels else np.array([])
    if indices is None:
        names = [os.path.basename(fp) for fp in dataset.filepaths]
    else:
        names = [os.path.basename(dataset.filepaths[i]) for i in indices]
    return probs, labels, names


def run_inference_onnx(onnx_path: str,
                       dataset: TensorAudioDataset,
                       batch_size: int = 1,
                       indices: Optional[List[int]] = None,
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Same semantics as run_inference but through an onnxruntime session.

    batch_size is forced to 1 because the ONNX we export has static shape
    (batch=1) to enable static INT8 PTQ (Step 5).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path,
                                providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    ds = Subset(dataset, indices) if indices is not None else dataset
    probs, labels = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = sess.run(None, {in_name: x.numpy().astype(np.float32)})[0]
        p = float(np.exp(out[0, 1]) / (np.exp(out[0, 0]) + np.exp(out[0, 1])))
        probs.append(p)
        labels.append(int(y))
    if indices is None:
        names = [os.path.basename(fp) for fp in dataset.filepaths]
    else:
        names = [os.path.basename(dataset.filepaths[i]) for i in indices]
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64), names


# ──────────────────────────────────────────────────────────────────────
# Phase A — every numeric row in one dict, used by verify_performance_claims
# ──────────────────────────────────────────────────────────────────────

def compute_phase_a_metrics(probs: np.ndarray,
                            labels: np.ndarray,
                            names: List[str],
                            threshold: float = 0.5
                            ) -> Dict[str, object]:
    """Return every load-bearing Phase A number as a plain dict.

    The returned dict is the sole source for Phase A rows in
    verify_performance_claims.py and for the JSON the pitch-figure script
    now consumes (Step 9). No caller should recompute any of these numbers.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    auc = float(roc_auc_score(labels, probs))
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    f1w = float(f1_score(labels, preds, average='weighted'))

    is_pos = labels == 1
    is_neg = labels == 0
    overall_tpr = float((probs[is_pos] >= threshold).mean()) if is_pos.any() else 0.0
    overall_fpr = float((probs[is_neg] >= threshold).mean()) if is_neg.any() else 0.0

    by_source: Dict[Tuple[int, str], List[float]] = {}
    for p, y, n in zip(probs, labels, names):
        by_source.setdefault((int(y), source_for(n)), []).append(float(p))

    tpr_by_source: Dict[str, Dict[str, float]] = {}
    fpr_by_source: Dict[str, Dict[str, float]] = {}
    for (y, src), ps in by_source.items():
        arr = np.asarray(ps)
        row = {'n': int(len(arr)),
               'rate': float((arr >= threshold).mean())}
        if y == 1:
            tpr_by_source[src] = row
        else:
            fpr_by_source[src] = row

    # Confusion matrix at the given threshold, normalized row-wise (as the
    # pitch-figure script expects).
    tp = int(((labels == 1) & (probs >= threshold)).sum())
    fn = int(((labels == 1) & (probs <  threshold)).sum())
    fp = int(((labels == 0) & (probs >= threshold)).sum())
    tn = int(((labels == 0) & (probs <  threshold)).sum())
    n_pos = tp + fn
    n_neg = fp + tn
    cm_normalized = [
        [tn / n_neg if n_neg else 0.0, fp / n_neg if n_neg else 0.0],
        [fn / n_pos if n_pos else 0.0, tp / n_pos if n_pos else 0.0],
    ]

    # Threshold sweep for the ROC-curve-style rows used in figures.
    sweep = []
    for t in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        pred_t = probs >= t
        fpr_t = float(pred_t[is_neg].mean()) if is_neg.any() else 0.0
        tpr_t = float(pred_t[is_pos].mean()) if is_pos.any() else 0.0
        bal = 0.5 * (tpr_t + (1 - fpr_t))
        sweep.append({'thr': t, 'fpr': fpr_t, 'tpr': tpr_t, 'bal_acc': bal})

    return {
        'threshold': threshold,
        'n_samples': int(len(labels)),
        'n_safe': int(is_neg.sum()),
        'n_unsafe': int(is_pos.sum()),
        'auc_roc': auc,
        'accuracy': acc,
        'f1_weighted': f1w,
        'overall_tpr': overall_tpr,
        'overall_fpr': overall_fpr,
        'tpr_by_source': tpr_by_source,
        'fpr_by_source': fpr_by_source,
        'confusion_matrix_normalized': cm_normalized,
        'confusion_matrix_counts': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'threshold_sweep': sweep,
    }


# ──────────────────────────────────────────────────────────────────────
# Latency timing
# ──────────────────────────────────────────────────────────────────────

def time_forward(fn: Callable,
                 x,
                 n_warmup: int = 30,
                 n_measure: int = 100
                 ) -> Dict[str, float]:
    """Return {'median', 'p10', 'p90', 'p99', 'mean', 'std'} ms for calling fn(x)."""
    for _ in range(n_warmup):
        fn(x)
    samples = np.empty(n_measure, dtype=np.float64)
    for i in range(n_measure):
        t0 = time.perf_counter()
        fn(x)
        samples[i] = (time.perf_counter() - t0) * 1000
    return {
        'median': float(np.median(samples)),
        'p10': float(np.percentile(samples, 10)),
        'p90': float(np.percentile(samples, 90)),
        'p99': float(np.percentile(samples, 99)),
        'mean': float(samples.mean()),
        'std': float(samples.std()),
        'n_samples': int(n_measure),
    }


# ──────────────────────────────────────────────────────────────────────
# Hardware disclosure — every latency report must emit one
# ──────────────────────────────────────────────────────────────────────

def hw_disclosure() -> Dict[str, object]:
    """Snapshot CPU / BLAS / version info for latency report footer."""
    cpu_model = _first_match(
        ['lscpu'], prefix='Model name:')
    governor = _first_match(
        ['bash', '-c',
         "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown"])
    torch_version = torch.__version__
    try:
        import onnxruntime as _ort
        ort_version = _ort.__version__
    except Exception:
        ort_version = None
    try:
        import scipy, librosa, numpy as _np
        scipy_version = scipy.__version__
        librosa_version = librosa.__version__
        numpy_version = _np.__version__
    except Exception:
        scipy_version = librosa_version = numpy_version = None

    blas = 'mkl' if torch.backends.mkl.is_available() else 'openblas-or-other'
    return {
        'cpu_model': cpu_model.strip() if cpu_model else 'unknown',
        'arch': _first_match(['uname', '-m']).strip(),
        'cores': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'governor': governor.strip() if governor else 'unknown',
        'torch_version': torch_version,
        'onnxruntime_version': ort_version,
        'scipy_version': scipy_version,
        'librosa_version': librosa_version,
        'numpy_version': numpy_version,
        'blas': blas,
        'target_hw_env': os.environ.get('TARGET_HW', 'unset'),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'unset'),
        'mkl_num_threads': os.environ.get('MKL_NUM_THREADS', 'unset'),
    }


def _first_match(cmd: List[str], prefix: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ''
    if prefix is None:
        return out
    for line in out.splitlines():
        if line.strip().startswith(prefix):
            return line.split(':', 1)[1].strip()
    return ''


# ──────────────────────────────────────────────────────────────────────
# Model size
# ──────────────────────────────────────────────────────────────────────

def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_stats(path: str = MODEL_SAVE_PATH) -> Dict[str, object]:
    """{'params': int, 'fp32_mb': float, 'int8_mb': float|None, ...}."""
    model = load_model(path, torch.device('cpu'))
    out = {
        'path': path,
        'params': param_count(model),
        'fp32_mb': file_size_mb(path),
    }
    onnx_path = path.replace('.pth', '.onnx')
    out['onnx_mb'] = file_size_mb(onnx_path) if os.path.exists(onnx_path) else None
    int8_pth = path.replace('.pth', '_int8.pth')
    out['int8_pth_mb'] = file_size_mb(int8_pth) if os.path.exists(int8_pth) else None
    int8_onnx = path.replace('.pth', '_int8.onnx')
    out['int8_onnx_mb'] = file_size_mb(int8_onnx) if os.path.exists(int8_onnx) else None
    return out
