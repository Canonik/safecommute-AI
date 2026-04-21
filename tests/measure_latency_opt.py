"""
Second-pass latency optimization — techniques that actually touch the
CNN blocks (which dominate safecommute_v2's compute, not the GRU/FC
that dynamic-INT8 hit in the previous sweep).

Configs tried:
  - ONNX Runtime FP32, ORT_ENABLE_ALL graph opts
  - ONNX Runtime dynamic INT8 (via onnxruntime.quantization.quantize_dynamic)
  - ONNX Runtime static INT8, calibrated on real test spectrograms
    (this is the one that quantizes conv layers)
  - torch.compile (inductor backend, inference-mode)

Run:  PYTHONPATH=. python -u tests/measure_latency_opt.py
"""

import os
import sys
import time
import tempfile
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, N_MELS, TIME_FRAMES, DATA_DIR

warnings.filterwarnings('ignore')

N_WARMUP = 30
N_MEASURE = 80


def time_it(fn, x, label):
    print(f"  running: {label:<40}", end=' ', flush=True)
    t0 = time.perf_counter()
    for _ in range(N_WARMUP):
        fn(x)
    warm = time.perf_counter() - t0
    samples = np.empty(N_MEASURE)
    for i in range(N_MEASURE):
        t1 = time.perf_counter()
        fn(x)
        samples[i] = (time.perf_counter() - t1) * 1000
    med = float(np.median(samples))
    p10 = float(np.percentile(samples, 10))
    p90 = float(np.percentile(samples, 90))
    print(f"median={med:7.2f}ms  p10={p10:7.2f}ms  p90={p90:7.2f}ms  "
          f"(warm {warm:.1f}s)")
    return med


def export_onnx(model, x, path):
    with torch.inference_mode():
        torch.onnx.export(
            model, x, path,
            input_names=['pcen'], output_names=['logits'],
            opset_version=17, dynamic_axes=None,
        )


def load_calibration_tensors(n=50):
    """Load ~n real test spectrograms for static-quant calibration."""
    test_safe = os.path.join(DATA_DIR, 'test', '0_safe')
    test_unsafe = os.path.join(DATA_DIR, 'test', '1_unsafe')
    tensors = []
    for d in (test_safe, test_unsafe):
        files = sorted(f for f in os.listdir(d)
                       if f.endswith('.pt') and not f.endswith('_teacher.pt'))
        for f in files[: n // 2]:
            t = torch.load(os.path.join(d, f), weights_only=True)
            # enforce (1, 64, 188) → batch-ready (1, 1, 64, 188)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            # enforce time dim
            if t.shape[-1] > TIME_FRAMES:
                t = t[..., :TIME_FRAMES]
            elif t.shape[-1] < TIME_FRAMES:
                pad = torch.zeros(*t.shape[:-1], TIME_FRAMES - t.shape[-1])
                t = torch.cat([t, pad], dim=-1)
            tensors.append(t.numpy().astype(np.float32))
    return tensors


def main():
    print(f"Torch: {torch.__version__}  threads={torch.get_num_threads()}")
    print()

    model = SafeCommuteCNN()
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()
    x = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    np_x = x.numpy()
    threads = torch.get_num_threads()

    # ── ONNX export (shared by the three ORT configs) ────────────────────
    onnx_fp32 = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name
    export_onnx(model, x, onnx_fp32)

    # ── ORT FP32 with all graph optimizations enabled ────────────────────
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_fp32, sess_options=opts,
                                providers=['CPUExecutionProvider'])
    def run_ort_fp32(_x):
        sess.run(None, {'pcen': np_x})
    time_it(run_ort_fp32, x, f"ORT FP32 + ENABLE_ALL, {threads}T")

    # ── ORT dynamic INT8 quantization ────────────────────────────────────
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static, QuantType, CalibrationDataReader,
    )

    onnx_dyn = onnx_fp32.replace('.onnx', '.dyn.onnx')
    try:
        quantize_dynamic(onnx_fp32, onnx_dyn,
                         weight_type=QuantType.QInt8)
        sess_dyn = ort.InferenceSession(onnx_dyn, sess_options=opts,
                                        providers=['CPUExecutionProvider'])
        def run_ort_dyn(_x):
            sess_dyn.run(None, {'pcen': np_x})
        time_it(run_ort_dyn, x, f"ORT INT8 dynamic, {threads}T")
    except Exception as e:
        print(f"  ORT dynamic quant FAILED: {type(e).__name__}: {e}")

    # ── ORT static INT8 quantization with real calibration data ──────────
    onnx_stat = onnx_fp32.replace('.onnx', '.stat.onnx')
    try:
        calib_arrays = load_calibration_tensors(n=50)
        print(f"  (loaded {len(calib_arrays)} calibration spectrograms)")

        class PTQReader(CalibrationDataReader):
            def __init__(self, arrays):
                self._it = iter([{'pcen': a} for a in arrays])
            def get_next(self):
                return next(self._it, None)

        # ORT's quantize_static needs a pre-processed model — the preprocess
        # inserts QDQ-friendly shapes. Running it keeps conv ops quantizable.
        onnx_prep = onnx_fp32.replace('.onnx', '.prep.onnx')
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(onnx_fp32, onnx_prep, skip_symbolic_shape=False)

        quantize_static(
            onnx_prep, onnx_stat,
            calibration_data_reader=PTQReader(calib_arrays),
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        sess_stat = ort.InferenceSession(onnx_stat, sess_options=opts,
                                         providers=['CPUExecutionProvider'])
        def run_ort_stat(_x):
            sess_stat.run(None, {'pcen': np_x})
        time_it(run_ort_stat, x, f"ORT INT8 static (calibrated), {threads}T")
    except Exception as e:
        print(f"  ORT static quant FAILED: {type(e).__name__}: {e}")

    # ── torch.compile ────────────────────────────────────────────────────
    try:
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        compiled = torch.compile(model, mode='reduce-overhead')
        def run_compile(_x):
            with torch.inference_mode():
                compiled(_x)
        time_it(run_compile, x, f"torch.compile reduce-overhead, {threads}T")
    except Exception as e:
        print(f"  torch.compile FAILED: {type(e).__name__}: {e}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    for p in (onnx_fp32, onnx_dyn, onnx_stat,
              onnx_fp32.replace('.onnx', '.prep.onnx')):
        try:
            os.unlink(p)
        except OSError:
            pass


if __name__ == '__main__':
    main()
