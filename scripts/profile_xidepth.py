import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet


def compute_conv_macs(module: nn.Conv2d, input_shape, output_shape) -> int:
    # input_shape: (B, C_in, H_in, W_in)
    # output_shape: (B, C_out, H_out, W_out)
    batch_size, out_c, out_h, out_w = output_shape
    in_c = input_shape[1]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_c // module.groups)
    return int(out_h * out_w * out_c * kernel_ops)


def calculate_analytical_macs(model: nn.Module, input_tensor: torch.Tensor):
    total_macs = 0
    stage_macs = {"stem": 0, "stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0, "decoder": 0, "heads": 0}
    hooks = []

    def make_hook(name):
        def hook(m, inp, outp):
            nonlocal total_macs
            if isinstance(m, nn.Conv2d):
                m_macs = compute_conv_macs(m, inp[0].shape, outp.shape)
                total_macs += m_macs
                for stage_key in stage_macs:
                    if stage_key in name:
                        stage_macs[stage_key] += m_macs
                        break
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    return total_macs, stage_macs


def profile_network(model_name: str, model: nn.Module, input_tensor: torch.Tensor, warmup: int = 10, runs: int = 30):
    model.eval()
    print(f"\n{'='*70}\n Profiling: {model_name} (Input: {list(input_tensor.shape)})\n{'='*70}")

    # 1. Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 2. Timing
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Overall Latency: {mean_lat:.2f} ms (P95: {p95_lat:.2f} ms) | Throughput: {fps:.2f} FPS")
    print(f"Parameters: {total_params:.2f} M")

    # 3. Analytical MACs & FLOPs
    macs, stage_macs = calculate_analytical_macs(model, input_tensor)
    gmacs = macs / 1e9
    gflops = (macs * 2) / 1e9
    print(f"Computational Complexity: {gmacs:.2f} GMACs ({gflops:.2f} GFLOPs)")

    # 4. PyTorch Profiler: Operator & Type breakdown
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            with record_function("model_inference"):
                _ = model(input_tensor)

    print("\n--- Top 10 Operators by CPU Time ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Aggregate operator categories
    op_table = prof.key_averages()
    cat_time = 0.0
    conv_time = 0.0
    shuffle_time = 0.0
    upsample_time = 0.0
    activation_time = 0.0
    norm_time = 0.0
    other_time = 0.0

    for item in op_table:
        name = item.key
        t_cpu = item.cpu_time_total
        if "conv" in name.lower() or "mkldnn_convolution" in name.lower():
            conv_time += t_cpu
        elif "cat" in name.lower() or "concat" in name.lower():
            cat_time += t_cpu
        elif "reshape" in name.lower() or "transpose" in name.lower() or "permute" in name.lower() or "view" in name.lower() or "slice" in name.lower():
            shuffle_time += t_cpu
        elif "upsample" in name.lower() or "interpolate" in name.lower():
            upsample_time += t_cpu
        elif "relu" in name.lower() or "sigmoid" in name.lower():
            activation_time += t_cpu
        elif "batch_norm" in name.lower():
            norm_time += t_cpu
        else:
            other_time += t_cpu

    total_op_time = conv_time + cat_time + shuffle_time + upsample_time + activation_time + norm_time + other_time
    if total_op_time > 0:
        print("\n--- Operator Category Share of CPU Latency ---")
        print(f"  Convolutions:        {conv_time / 1000.0:8.2f} ms ({conv_time / total_op_time * 100:5.1f}%)")
        print(f"  Concatenations:      {cat_time / 1000.0:8.2f} ms ({cat_time / total_op_time * 100:5.1f}%)")
        print(f"  Channel Shuffle/Reshape: {shuffle_time / 1000.0:4.2f} ms ({shuffle_time / total_op_time * 100:5.1f}%)")
        print(f"  Upsampling/Interp:   {upsample_time / 1000.0:8.2f} ms ({upsample_time / total_op_time * 100:5.1f}%)")
        print(f"  Activations:         {activation_time / 1000.0:8.2f} ms ({activation_time / total_op_time * 100:5.1f}%)")
        print(f"  Batch Normalization: {norm_time / 1000.0:8.2f} ms ({norm_time / total_op_time * 100:5.1f}%)")
        print(f"  Other / Overhead:    {other_time / 1000.0:8.2f} ms ({other_time / total_op_time * 100:5.1f}%)")

    return {
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "fps": fps,
        "params_m": total_params,
        "gmacs": gmacs,
        "gflops": gflops,
        "conv_pct": conv_time / total_op_time * 100 if total_op_time > 0 else 0,
        "cat_pct": cat_time / total_op_time * 100 if total_op_time > 0 else 0,
        "shuffle_pct": shuffle_time / total_op_time * 100 if total_op_time > 0 else 0,
        "upsample_pct": upsample_time / total_op_time * 100 if total_op_time > 0 else 0,
    }


def main():
    torch.set_num_threads(1)  # Profile single-thread to isolate true algorithmic work
    print(f"Profiling environment: Single-thread CPU (isolated algorithmic profiling)")

    dummy_input = torch.randn(1, 3, 192, 640)

    xi = XiDepthNet(num_scales=4)
    res = ResNetDepthNet(num_scales=4, pretrained=False)

    xi_results = profile_network("XiDepthNet (2.36M Params, XiBlock)", xi, dummy_input)
    res_results = profile_network("ResNetDepthNet (14.72M Params, ResNet-18)", res, dummy_input)

    print("\n" + "=" * 70)
    print(" Comparative Architectural Profile Summary (Single-Thread CPU)")
    print("=" * 70)
    print(f"{'Metric':<28} | {'XiDepthNet':>16} | {'ResNetDepthNet':>16} | {'Ratio':>6}")
    print("-" * 70)
    print(f"{'Parameters (M)':<28} | {xi_results['params_m']:16.2f} | {res_results['params_m']:16.2f} | {res_results['params_m']/xi_results['params_m']:5.1f}x")
    print(f"{'Complexity (GMACs)':<28} | {xi_results['gmacs']:16.2f} | {res_results['gmacs']:16.2f} | {res_results['gmacs']/xi_results['gmacs']:5.1f}x")
    print(f"{'Mean Latency (ms)':<28} | {xi_results['mean_latency_ms']:16.2f} | {res_results['mean_latency_ms']:16.2f} | {res_results['mean_latency_ms']/xi_results['mean_latency_ms']:5.2f}x")
    print(f"{'Throughput (FPS)':<28} | {xi_results['fps']:16.2f} | {res_results['fps']:16.2f} | {xi_results['fps']/res_results['fps']:5.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
