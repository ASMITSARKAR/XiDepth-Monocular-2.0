import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet, OfficialMonodepth2


def get_conv_type(m: nn.Conv2d):
    if m.groups == m.in_channels and m.groups == m.out_channels and m.groups > 1:
        return "depthwise"
    elif m.kernel_size == (1, 1):
        return "pointwise"
    else:
        return "dense"


def benchmark_model(model: nn.Module, input_tensor: torch.Tensor, num_threads: int, num_runs: int = 100):
    torch.set_num_threads(num_threads)
    model.eval()

    # Warmup (discarded)
    with torch.no_grad():
        for _ in range(15):
            _ = model(input_tensor)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = 1000.0 / median_lat
    return {
        "median": median_lat,
        "p95": p95_lat,
        "mean": mean_lat,
        "std": std_lat,
        "fps": fps,
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }


def analyze_model_macs(model: nn.Module, input_tensor: torch.Tensor):
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    total_macs = flops.total()  # fvcore counts multiply-accumulates (MACs)
    total_gmacs = total_macs / 1e9

    conv_stats = {
        "depthwise": {"macs": 0, "params": 0, "count": 0},
        "pointwise": {"macs": 0, "params": 0, "count": 0},
        "dense":     {"macs": 0, "params": 0, "count": 0}
    }

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            ctype = get_conv_type(m)
            m_params = sum(p.numel() for p in m.parameters())
            m_macs = flops.by_module().get(name, 0)
            conv_stats[ctype]["macs"] += m_macs
            conv_stats[ctype]["params"] += m_params
            conv_stats[ctype]["count"] += 1

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    return total_gmacs, total_params, conv_stats


def main():
    dummy = torch.randn(1, 3, 192, 640)
    models_dict = {
        "OfficialMonodepth2": OfficialMonodepth2(num_scales=4, pretrained=False),
        "XiDepthNet": XiDepthNet(num_scales=4),
        "ResNetDepthNet": ResNetDepthNet(num_scales=4, pretrained=False),
    }

    results = {}

    print("=" * 80)
    print(" 100-RUN EMPIRICAL BENCHMARK (Mean ± Std over 100 Runs, Batch Size 1, 192x640)")
    print("=" * 80)

    # 1. Complexity (GMACs & Params)
    for name, model in models_dict.items():
        gmacs, params, conv_stats = analyze_model_macs(model, dummy)
        results[name] = {"gmacs": gmacs, "params": params, "conv_stats": conv_stats}

    # 2. Single-Thread Latency (Pinned 1 Thread)
    print("\n--- Running Single-Thread Benchmark (1 Thread, 100 Runs) ---")
    for name, model in models_dict.items():
        bench = benchmark_model(model, dummy, num_threads=1, num_runs=100)
        results[name]["st"] = bench
        print(f"  {name:<20}: Median {bench['median']:6.2f} ms | P95 {bench['p95']:6.2f} ms | Mean {bench['mean']:6.2f} ± {bench['std']:5.2f} ms ({bench['fps']:5.2f} FPS)")

    # 3. Multi-Thread Latency (All Host Threads)
    host_threads = os.cpu_count() or 16
    print(f"\n--- Running Multi-Thread Benchmark ({host_threads} Threads, 100 Runs) ---")
    for name, model in models_dict.items():
        bench = benchmark_model(model, dummy, num_threads=host_threads, num_runs=100)
        results[name]["mt"] = bench
        print(f"  {name:<20}: Median {bench['median']:6.2f} ms | P95 {bench['p95']:6.2f} ms | Mean {bench['mean']:6.2f} ± {bench['std']:5.2f} ms ({bench['fps']:5.2f} FPS)")

    # Print markdown table
    print("\n" + "=" * 105)
    print(" FINAL SUMMARY TABLE (Batch Size 1, 192x640, 100 Runs, 15 Warmup Discarded)")
    print("=" * 105)
    print("| Architecture | Params | GMACs | Single-Thread Median (P95) [Mean±Std] | Multi-Thread Median (P95) [Mean±Std] |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for name, r in results.items():
        st = r["st"]
        mt = r["mt"]
        print(
            f"| **{name}** | {r['params']:.2f} M | {r['gmacs']:.2f} G | "
            f"**{st['median']:.1f} ms** ({st['p95']:.1f} ms) [{st['mean']:.1f}±{st['std']:.1f} ms] | "
            f"**{mt['median']:.1f} ms** ({mt['p95']:.1f} ms) [{mt['mean']:.1f}±{mt['std']:.1f} ms] |"
        )


if __name__ == "__main__":
    main()
