import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import XiDepthNet, ResNetDepthNet, disp_to_depth


def compute_depth_errors(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Evaluation running on: {device}")

    # Model initialization
    if args.model == "xidepth":
        model = XiDepthNet(num_scales=4)
    elif args.model == "resnet18":
        model = ResNetDepthNet(num_scales=4, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["depth_net"] if "depth_net" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        print("Warning: Evaluating model without checkpoint weights (random / init weights)")

    model.to(device)
    model.eval()

    with open(args.split_file, "r") as f:
        filenames = [l.strip() for l in f if l.strip()]

    # Load ground truth depth maps if available
    gt_depths = None
    if args.gt_path and os.path.isfile(args.gt_path):
        gt_data = np.load(args.gt_path, allow_pickle=True)
        gt_depths = gt_data["data"]
        print(f"Loaded {len(gt_depths)} ground truth depth maps from {args.gt_path}")

    errors: List[Tuple[float, ...]] = []

    print(f"Evaluating {len(filenames)} test images...")
    with torch.no_grad():
        for i, line in enumerate(filenames):
            parts = line.split()
            folder = parts[0]
            frame_idx = int(parts[1])
            side = parts[2] if len(parts) > 2 else "l"
            cam_dir = "image_02" if side == "l" else "image_03"
            frame_name = f"{frame_idx:010d}"

            # Candidate paths
            candidates = [
                os.path.join(args.dataset_dir, folder, cam_dir, "data", f"{frame_name}.png"),
                os.path.join(args.dataset_dir, folder, cam_dir, "data", f"{frame_name}.jpg"),
                os.path.join(args.dataset_dir, folder, cam_dir, f"{frame_name}.png"),
                os.path.join(args.dataset_dir, folder, cam_dir, f"{frame_name}.jpg"),
            ]
            img_path = None
            for p in candidates:
                if os.path.isfile(p):
                    img_path = p
                    break

            if img_path is None:
                img = Image.new("RGB", (1242, 375), color=(128, 128, 128))
            else:
                img = Image.open(img_path).convert("RGB")

            orig_w, orig_h = img.size
            img_tensor = img.resize((args.width, args.height), Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_tensor).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred_disp = model(img_tensor)
            if isinstance(pred_disp, list):
                pred_disp = pred_disp[0]

            _, pred_depth = disp_to_depth(pred_disp, args.min_depth, args.max_depth)
            pred_depth = pred_depth.cpu().numpy().squeeze()

            # Resize predicted depth back to original resolution
            pred_depth_resized = Image.fromarray(pred_depth).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            pred_depth_resized = np.array(pred_depth_resized)

            if gt_depths is not None and i < len(gt_depths):
                gt = gt_depths[i]

                # Standard Garg crop
                crop = np.array([0.4081 * orig_h, 0.9918 * orig_h, 0.0359 * orig_w, 0.9640 * orig_w]).astype(int)
                crop_mask = np.zeros(gt.shape, dtype=bool)
                crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = True

                valid_mask = (gt > args.min_depth) & (gt < args.max_depth_cap) & crop_mask

                if not valid_mask.any():
                    continue

                valid_gt = gt[valid_mask]
                valid_pred = pred_depth_resized[valid_mask]

                # Median scaling
                ratio = np.median(valid_gt) / np.median(valid_pred)
                valid_pred *= ratio
                valid_pred[valid_pred < args.min_depth] = args.min_depth
                valid_pred[valid_pred > args.max_depth_cap] = args.max_depth_cap

                errors.append(compute_depth_errors(valid_gt, valid_pred))

            if (i + 1) % 100 == 0:
                print(f"Evaluated [{i+1}/{len(filenames)}]")

    if not errors:
        print("No valid evaluation pairs could be evaluated against ground truth.")
        return

    mean_errors = np.array(errors).mean(0)
    results = {
        "abs_rel": float(mean_errors[0]),
        "sq_rel": float(mean_errors[1]),
        "rmse": float(mean_errors[2]),
        "rmse_log": float(mean_errors[3]),
        "a1": float(mean_errors[4]),
        "a2": float(mean_errors[5]),
        "a3": float(mean_errors[6]),
    }

    print("\n" + "=" * 65)
    print(" KITTI Eigen Benchmark Results")
    print("=" * 65)
    print(f"{'Abs Rel':>10} | {'Sq Rel':>10} | {'RMSE':>10} | {'RMSE log':>10} | {'d < 1.25':>10} | {'d < 1.25^2':>10}")
    print("-" * 65)
    print(
        f"{results['abs_rel']:10.4f} | {results['sq_rel']:10.4f} | {results['rmse']:10.4f} | "
        f"{results['rmse_log']:10.4f} | {results['a1']:10.4f} | {results['a2']:10.4f}"
    )
    print("=" * 65)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XiDepth models on KITTI Eigen test benchmark")
    parser.add_argument("--model", type=str, default="xidepth", choices=["xidepth", "resnet18"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="data/kitti")
    parser.add_argument("--split_file", type=str, default="data/splits/eigen_zhou/test_files.txt")
    parser.add_argument("--gt_path", type=str, default="data/gt_depths.npz")
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--max_depth_cap", type=float, default=80.0)
    parser.add_argument("--no_cuda", action="store_true", default=False)

    args = parser.parse_args()
    evaluate(args)
