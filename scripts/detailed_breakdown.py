import sys
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet, OfficialMonodepth2
from fvcore.nn import FlopCountAnalysis, flop_count_table
from ptflops import get_model_complexity_info


def analyze_model(name, model, input_size=(1, 3, 192, 640)):
    model.eval()
    dummy = torch.randn(*input_size)
    print(f"\n{'='*75}\n MODEL ANALYSIS: {name}\n{'='*75}")

    # fvcore analysis
    flops = FlopCountAnalysis(model, dummy)
    total_flops = flops.total()
    total_gflops = total_flops / 1e9
    total_gmacs = total_gflops / 2.0  # 1 MAC = 2 FLOPs

    print(f"fvcore Total FLOPs: {total_gflops:.3f} GFLOPs ({total_gmacs:.3f} GMACs)")

    # ptflops analysis
    macs_pt, params_pt = get_model_complexity_info(
        model, (3, 192, 640), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    print(f"ptflops Total MACs: {macs_pt / 1e9:.3f} GMACs | Params: {params_pt / 1e6:.3f} M")

    print("\n--- fvcore Module-Level FLOP Table (Top-level + Depth 2) ---")
    print(flop_count_table(flops, max_depth=2))

    return flops, total_gmacs


def main():
    xi = XiDepthNet(num_scales=4)
    res = ResNetDepthNet(num_scales=4, pretrained=False)
    try:
        off = OfficialMonodepth2()
    except Exception as e:
        off = None
        print("OfficialMonodepth2 could not be instantiated:", e)

    analyze_model("XiDepthNet", xi)
    analyze_model("ResNetDepthNet", res)
    if off:
        analyze_model("OfficialMonodepth2", off)


if __name__ == "__main__":
    main()
