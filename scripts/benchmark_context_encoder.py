"""MCR-UAV Phase 6: Forward-Pass Latency Benchmark for Transformer Context Encoder."""

import os
import sys
import time
import numpy as np
import torch

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_context_encoder import TransformerContextEncoder


def benchmark_device(device_name: str, num_warmup: int = 50, num_iters: int = 500) -> dict:
    device = torch.device(device_name)
    model = TransformerContextEncoder().to(device)
    model.eval()

    results = {}
    for batch_size in [1, 32]:
        x = torch.randn(batch_size, 20, 52, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed benchmark iterations
        latencies_ms = []
        with torch.no_grad():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        std_lat = float(np.std(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))

        results[batch_size] = {
            "mean_ms": round(mean_lat, 4),
            "std_ms": round(std_lat, 4),
            "median_ms": round(median_lat, 4),
            "p95_ms": round(p95_lat, 4),
            "p99_ms": round(p99_lat, 4),
            "min_ms": round(min_lat, 4),
            "max_ms": round(max_lat, 4),
        }

    return results


def main():
    print("=" * 80)
    print("MCR-UAV TRANSFORMER CONTEXT ENCODER LATENCY BENCHMARK")
    print("=" * 80)

    model = TransformerContextEncoder()
    param_count = model.count_parameters()
    print(f"Model Configuration:")
    print(f"  Input Dim (d_h):    {model.input_dim}")
    print(f"  Sequence Len (L):   {model.seq_len}")
    print(f"  Embedding Dim (d):  {model.d_model}")
    print(f"  Attention Heads:    {model.n_heads}")
    print(f"  Encoder Layers:     {model.n_layers}")
    print(f"  Latent Dim (d_z):   {model.latent_dim}")
    print(f"  Trainable Params:   {param_count:,}")
    print("-" * 80)

    # Benchmark on CPU
    print("[Benchmarking on CPU...]")
    cpu_res = benchmark_device("cpu")
    print(f"CPU Latency (B=1, online control step):")
    print(f"  Mean:   {cpu_res[1]['mean_ms']:.4f} ms (+/- {cpu_res[1]['std_ms']:.4f} ms)")
    print(f"  Median: {cpu_res[1]['median_ms']:.4f} ms")
    print(f"  P95:    {cpu_res[1]['p95_ms']:.4f} ms")
    print(f"  P99:    {cpu_res[1]['p99_ms']:.4f} ms")
    print(f"CPU Latency (B=32, batch inference):")
    print(f"  Mean:   {cpu_res[32]['mean_ms']:.4f} ms | P95: {cpu_res[32]['p95_ms']:.4f} ms")

    # Benchmark on CUDA if available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"\n[Benchmarking on CUDA: {device_name}...]")
        cuda_res = benchmark_device("cuda")
        print(f"CUDA Latency (B=1, online control step):")
        print(f"  Mean:   {cuda_res[1]['mean_ms']:.4f} ms (+/- {cuda_res[1]['std_ms']:.4f} ms)")
        print(f"  Median: {cuda_res[1]['median_ms']:.4f} ms")
        print(f"  P95:    {cuda_res[1]['p95_ms']:.4f} ms")
        print(f"  P99:    {cuda_res[1]['p99_ms']:.4f} ms")
        print(f"CUDA Latency (B=32, batch inference):")
        print(f"  Mean:   {cuda_res[32]['mean_ms']:.4f} ms | P95: {cuda_res[32]['p95_ms']:.4f} ms")
    else:
        print("\n[CUDA not available in this environment]")

    print("=" * 80)


if __name__ == "__main__":
    main()
