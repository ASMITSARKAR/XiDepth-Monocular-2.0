import argparse
import glob
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet, disp_to_depth


def visualize_image(model, img_path: str, output_path: str, device, height: int = 192, width: int = 640):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    img_tensor = img.resize((width, height), Image.Resampling.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_tensor).transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        disp = model(img_tensor)
        if isinstance(disp, list):
            disp = disp[0]
        _, depth = disp_to_depth(disp)
        depth_np = depth.cpu().numpy().squeeze()

    depth_resized = Image.fromarray(depth_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    depth_resized = np.array(depth_resized)

    # Invert disparity for standard visual representation (closer = brighter)
    disp_vis = 1.0 / np.clip(depth_resized, 0.1, 80.0)
    disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-7)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im = axes[1].imshow(disp_vis, cmap="magma")
    axes[1].set_title("Predicted Depth Map (Magma Colormap)")
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth visualization for RGB images")
    parser.add_argument("--model", type=str, default="xidepth", choices=["xidepth", "resnet18"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.model == "xidepth":
        model = XiDepthNet(num_scales=4)
    else:
        model = ResNetDepthNet(num_scales=4, pretrained=False)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["depth_net"] if "depth_net" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    if os.path.isdir(args.input):
        images = sorted(glob.glob(os.path.join(args.input, "*.png")) + glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        images = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    for img_path in images:
        out_name = f"depth_{Path(img_path).stem}.png"
        out_path = os.path.join(args.output_dir, out_name)
        visualize_image(model, img_path, out_path, device)


if __name__ == "__main__":
    main()
