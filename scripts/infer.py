import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet


def benchmark_model(model, device, height: int = 192, width: int = 640, num_runs: int = 50, warmup: int = 10):
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, height, width, device=device)

    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_ms = np.mean(latencies)
    median_ms = np.median(latencies)
    p95_ms = np.percentile(latencies, 95)
    fps = 1000.0 / mean_ms

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU inference latency and FPS")
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Benchmarking on device: {device}")

    xi = XiDepthNet(num_scales=4)
    res = ResNetDepthNet(num_scales=4, pretrained=False)

    xi_params = sum(p.numel() for p in xi.parameters()) / 1e6
    res_params = sum(p.numel() for p in res.parameters()) / 1e6

    print("\nBenchmarking XiDepthNet (XiBlock backbone)...")
    xi_perf = benchmark_model(xi, device, args.height, args.width, args.num_runs, args.warmup)

    print("Benchmarking ResNetDepthNet (ResNet-18 baseline)...")
    res_perf = benchmark_model(res, device, args.height, args.width, args.num_runs, args.warmup)

    print("\n" + "=" * 70)
    print(" Inference Latency & Efficiency Comparison")
    print("=" * 70)
    print(f"{'Model':<18} | {'Params':>8} | {'Mean Latency':>13} | {'P95 Latency':>12} | {'FPS':>8}")
    print("-" * 70)
    print(
        f"{'XiDepthNet':<18} | {xi_params:7.2f}M | {xi_perf['mean_ms']:10.2f} ms | "
        f"{xi_perf['p95_ms']:9.2f} ms | {xi_perf['fps']:8.2f}"
    )
    print(
        f"{'ResNetDepthNet':<18} | {res_params:7.2f}M | {res_perf['mean_ms']:10.2f} ms | "
        f"{res_perf['p95_ms']:9.2f} ms | {res_perf['fps']:8.2f}"
    )
    print("=" * 70)

    speedup = res_perf["mean_ms"] / xi_perf["mean_ms"]
    param_ratio = res_params / xi_params
    print(f"Summary: XiDepthNet is {param_ratio:.1f}x smaller and {speedup:.2f}x faster per frame on {device}.")


if __name__ == "__main__":
    main()
