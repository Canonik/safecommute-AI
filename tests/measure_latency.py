"""
Targeted CPU-latency benchmark for safecommute_v2, addressing the gap
between the documented ~12 ms claim and our 125 ms measurement on this
machine.

Variables swept:
  - eager + no_grad vs eager + inference_mode
  - thread count (1, 2, 4, default)
  - TorchScript trace + optimize_for_inference at 1 thread

Prints one row per config so any hang / slow config is visible live.
Run:  PYTHONPATH=. python -u tests/measure_latency.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, N_MELS, TIME_FRAMES

N_WARMUP = 30
N_MEASURE = 100  # 100 iterations × 125 ms worst case = 12.5 s / config


def time_it(fn, x, label):
    print(f"  running: {label:<30}", end=' ', flush=True)
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
          f"(warmup {warm:.1f}s)")
    return med, p10, p90


def main():
    print(f"Torch: {torch.__version__}  "
          f"threads={torch.get_num_threads()}  "
          f"interop={torch.get_num_interop_threads()}")
    print(f"MKL env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','-')}  "
          f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS','-')}")
    print()

    model = SafeCommuteCNN()
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()
    x = torch.randn(1, 1, N_MELS, TIME_FRAMES)

    orig_threads = torch.get_num_threads()

    def run_no_grad(_x):
        with torch.no_grad():
            model(_x)

    def run_inf(_x):
        with torch.inference_mode():
            model(_x)

    time_it(run_no_grad, x, f"eager+no_grad, {orig_threads}T")
    time_it(run_inf,     x, f"eager+inference_mode, {orig_threads}T")

    for t in (4, 2, 1):
        torch.set_num_threads(t)
        time_it(run_inf, x, f"inference_mode, {t}T")
    torch.set_num_threads(orig_threads)

    # TorchScript at 1 thread — best case for a tiny model.
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            scripted = torch.jit.trace(model, x)
            scripted = torch.jit.optimize_for_inference(scripted)

        def run_jit(_x):
            with torch.inference_mode():
                scripted(_x)

        time_it(run_jit, x, "jit+optimize, 1T")
    except Exception as e:
        print(f"  jit FAILED: {type(e).__name__}: {e}")
    finally:
        torch.set_num_threads(orig_threads)

    # ONNX Runtime — the most likely backend behind the "~12ms" claim
    # because onnx / onnxruntime are in requirements.txt.
    try:
        import onnxruntime as ort
        import tempfile

        torch.set_num_threads(orig_threads)
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
        model.eval()
        with torch.inference_mode():
            torch.onnx.export(
                model, x, onnx_path,
                input_names=['pcen'], output_names=['logits'],
                opset_version=17, dynamic_axes=None,
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = orig_threads
        sess = ort.InferenceSession(
            onnx_path, sess_options=sess_opts,
            providers=['CPUExecutionProvider'])
        np_x = x.numpy()

        def run_ort(_x):
            sess.run(None, {'pcen': np_x})

        time_it(run_ort, x, f"onnxruntime CPU, {orig_threads}T")

        sess_opts1 = ort.SessionOptions()
        sess_opts1.intra_op_num_threads = 1
        sess1 = ort.InferenceSession(
            onnx_path, sess_options=sess_opts1,
            providers=['CPUExecutionProvider'])
        def run_ort1(_x):
            sess1.run(None, {'pcen': np_x})
        time_it(run_ort1, x, "onnxruntime CPU, 1T")

        os.unlink(onnx_path)
    except Exception as e:
        print(f"  onnxruntime FAILED: {type(e).__name__}: {e}")

    # Dynamic INT8 quantization — typical 2–4× speedup on CPU.
    try:
        torch.set_num_threads(orig_threads)
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)
        qmodel.eval()
        def run_q(_x):
            with torch.inference_mode():
                qmodel(_x)
        time_it(run_q, x, f"dynamic INT8 quant, {orig_threads}T")
    except Exception as e:
        print(f"  quantize FAILED: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
