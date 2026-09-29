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


def profile_module_breakdown(model_name: str, model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100):
    model.eval()
    print(f"\n{'='*80}\n MODULE-LEVEL PROFILING: {model_name}\n{'='*80}")

    # Pinned single-thread execution
    torch.set_num_threads(1)

    # 1. Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(input_tensor)

    # 2. Overall Latency (Mean +- Std over 100 runs)
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    lat_mean = np.mean(latencies)
    lat_std = np.std(latencies)
    fps = 1000.0 / lat_mean

    # 3. FlopCountAnalysis from fvcore
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    total_gmacs = total_flops / (2.0 * 1e9)  # 1 MAC = 2 FLOPs

    # 4. Conv category breakdown (depthwise, pointwise, dense)
    conv_stats = {"depthwise": {"macs": 0, "params": 0, "count": 0},
                  "pointwise": {"macs": 0, "params": 0, "count": 0},
                  "dense":     {"macs": 0, "params": 0, "count": 0}}

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            ctype = get_conv_type(m)
            m_params = sum(p.numel() for p in m.parameters())
            # Compute MACs using hook or module flop
            # fvcore flop for this module:
            m_flops = flops.by_module().get(name, 0)
            m_macs = m_flops / 2.0
            conv_stats[ctype]["macs"] += m_macs
            conv_stats[ctype]["params"] += m_params
            conv_stats[ctype]["count"] += 1

    print(f"Overall Latency (100 runs, 1 thread): {lat_mean:.2f} ± {lat_std:.2f} ms ({fps:.2f} FPS)")
    print(f"Total Model Complexity: {total_gmacs:.3f} GMACs ({total_flops / 1e9:.3f} GFLOPs)")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    print("\n--- Convolution Type Breakdown ---")
    print(f"{'Type':<12} | {'Count':>6} | {'Params (M)':>12} | {'GMACs':>10} | {'% of MACs':>10}")
    print("-" * 60)
    for ctype, st in conv_stats.items():
        gmac = st["macs"] / 1e9
        pct = (st["macs"] / (total_flops / 2.0)) * 100 if total_flops > 0 else 0
        print(f"{ctype:<12} | {st['count']:>6} | {st['params']/1e6:>12.3f} | {gmac:>10.3f} | {pct:>9.1f}%")

    # 5. Encoder vs Decoder Breakdown
    enc_flops = 0
    dec_flops = 0
    head_flops = 0

    for mod_name, mod_flop in flops.by_module().items():
        # Check top-level submodule
        top = mod_name.split(".")[0]
        if top in ["conv1", "stage2", "stage3", "stage4", "stage5", "encoder"]:
            enc_flops += mod_flop
        elif top in ["upconv1", "upconv2", "upconv3", "upconv4", "upconv5", "iconv1", "iconv2", "iconv3", "iconv4", "iconv5", "decoder"]:
            dec_flops += mod_flop
        elif top in ["disp1", "disp2", "disp3", "disp4"]:
            head_flops += mod_flop

    # Note: by_module includes nested modules, so we use top-level by_module_and_operator or direct children
    top_breakdown = {}
    for name, child in model.named_children():
        child_flops = FlopCountAnalysis(child, input_tensor).total() if name in ["conv1", "encoder"] else None
    
    return {
        "lat_mean": lat_mean,
        "lat_std": lat_std,
        "fps": fps,
        "total_gmacs": total_gmacs,
        "conv_stats": conv_stats,
    }


def main():
    dummy = torch.randn(1, 3, 192, 640)
    xi = XiDepthNet(num_scales=4)
    res = ResNetDepthNet(num_scales=4, pretrained=False)
    off = OfficialMonodepth2()

    profile_module_breakdown("XiDepthNet", xi, dummy, num_runs=100)
    profile_module_breakdown("ResNetDepthNet", res, dummy, num_runs=100)
    profile_module_breakdown("OfficialMonodepth2", off, dummy, num_runs=100)


if __name__ == "__main__":
    main()
