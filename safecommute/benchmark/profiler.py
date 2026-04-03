"""Edge deployment profiling: model size, latency, parameter count."""

import io
import time

import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def measure_latency_ms(model, dummy_input, n_warmup=20, n_runs=200, device='cpu'):
    """Returns (mean_ms, std_ms, p99_ms)."""
    model.eval()
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy_input)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    return float(np.mean(times)), float(np.std(times)), float(np.percentile(times, 99))


def profile_model(model, dummy_input, device='cpu'):
    """Full profiling: params, size, latency."""
    model = model.to(device)
    model.eval()

    params = count_parameters(model)
    size_mb = measure_model_size_mb(model)
    mean_ms, std_ms, p99_ms = measure_latency_ms(model, dummy_input, device=device)

    return {
        'params': params,
        'params_human': f"{params / 1e6:.2f}M",
        'size_mb': round(size_mb, 2),
        'latency_mean_ms': round(mean_ms, 1),
        'latency_std_ms': round(std_ms, 1),
        'latency_p99_ms': round(p99_ms, 1),
    }
