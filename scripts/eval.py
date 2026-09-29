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

from models import XiDepthNet, ResNetDepthNet, OfficialMonodepth2, disp_to_depth


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


def batch_post_process_disparity(l_disp: np.ndarray, r_disp: np.ndarray) -> np.ndarray:
    """Official Monodepth blended post-processing."""
    h, w = l_disp.shape[-2:]
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = l_mask[:, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Evaluation running on: {device}")

    # Model initialization
    if args.model in ("monodepth2", "monodepth2_official"):
        model = OfficialMonodepth2()
        ckpt_dir = args.checkpoint if (args.checkpoint and os.path.isdir(args.checkpoint)) else "checkpoints/monodepth2_official"
        enc_path = os.path.join(ckpt_dir, "encoder.pth")
        dec_path = os.path.join(ckpt_dir, "depth.pth")
        if os.path.isfile(enc_path) and os.path.isfile(dec_path):
            model.load_pretrained(enc_path, dec_path)
            print(f"Loaded official Monodepth2 weights from: {ckpt_dir}")
        elif args.checkpoint and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict = checkpoint["depth_net"] if "depth_net" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        else:
            raise FileNotFoundError(
                f"Missing weights for monodepth2. Expected directory with encoder.pth/depth.pth "
                f"or checkpoint .pth file, got: {args.checkpoint}"
            )
    elif args.model == "xidepth":
        model = XiDepthNet(num_scales=4)
        if args.checkpoint and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict = checkpoint["depth_net"] if "depth_net" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        else:
            print("Warning: Evaluating model without checkpoint weights (random / init weights)")
    elif args.model == "resnet18":
        model = ResNetDepthNet(num_scales=4, pretrained=False)
        if args.checkpoint and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict = checkpoint["depth_net"] if "depth_net" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        else:
            print("Warning: Evaluating model without checkpoint weights (random / init weights)")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    model.eval()

    if not os.path.isfile(args.split_file):
        raise FileNotFoundError(f"Split file missing: {args.split_file}")

    with open(args.split_file, "r") as f:
        filenames = [l.strip() for l in f if l.strip()]

    # Load ground truth depth maps (mandatory hard check)
    if not args.gt_path or not os.path.isfile(args.gt_path):
        raise FileNotFoundError(f"Ground truth file missing: {args.gt_path}")

    gt_data = np.load(args.gt_path, allow_pickle=True, fix_imports=True, encoding="latin1")
    gt_depths = gt_data["data"]
    print(f"Loaded {len(gt_depths)} ground truth depth maps from {args.gt_path}")

    assert len(filenames) == len(gt_depths), (
        f"Alignment Error: split file has {len(filenames)} entries but GT has {len(gt_depths)} entries! "
        f"For improved GT (652 entries), use --split_file data/splits/eigen_benchmark/test_files.txt. "
        f"For raw GT (697 entries), use --split_file data/splits/eigen/test_files.txt."
    )

    is_benchmark = "benchmark" in os.path.basename(args.split_file).lower() or len(filenames) == 652
    print(f"Evaluation mode: {'Benchmark (improved GT, no crop)' if is_benchmark else 'Eigen (raw LiDAR GT, Garg crop)'}")

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
                raise FileNotFoundError(
                    f"Test frame {frame_name} in folder {folder} ({cam_dir}) not found under {args.dataset_dir}"
                )

            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img_tensor = img.resize((args.width, args.height), Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_tensor).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred_disp = model(img_tensor)
            if isinstance(pred_disp, (list, tuple)):
                pred_disp = pred_disp[0]

            if args.post_process:
                # Official blended post-processing
                img_flipped = torch.flip(img_tensor, dims=[3])
                pred_disp_flipped = model(img_flipped)
                if isinstance(pred_disp_flipped, (list, tuple)):
                    pred_disp_flipped = pred_disp_flipped[0]

                pred_disp_np = pred_disp.squeeze().cpu().numpy()
                pred_disp_flipped_np = pred_disp_flipped.squeeze().cpu().numpy()
                blended = batch_post_process_disparity(pred_disp_np, pred_disp_flipped_np[:, ::-1])
                pred_disp = torch.from_numpy(blended).unsqueeze(0).unsqueeze(0).to(device)

            # Convert to scaled disparity
            scaled_disp, _ = disp_to_depth(pred_disp, args.min_depth, args.max_depth)

            # Bilinear interpolation in disparity space to original image resolution
            scaled_disp = F.interpolate(scaled_disp, (orig_h, orig_w), mode="bilinear", align_corners=False)

            # Invert scaled disparity to obtain metric depth
            pred_depth = (1.0 / scaled_disp).squeeze().cpu().numpy()

            gt = gt_depths[i]
            assert pred_depth.shape == gt.shape, (
                f"Shape mismatch at index {i} ({filenames[i]}): pred {pred_depth.shape} != gt {gt.shape}"
            )

            # Split-aware crop and mask
            if not is_benchmark:
                # Standard Garg crop (exact Monodepth2 coefficients)
                crop = np.array([
                    0.40810811 * orig_h,
                    0.99189189 * orig_h,
                    0.03594771 * orig_w,
                    0.96405229 * orig_w,
                ]).astype(int)
                crop_mask = np.zeros(gt.shape, dtype=bool)
                crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = True
                valid_mask = (gt > args.eval_min_depth) & (gt < args.eval_max_depth) & crop_mask
            else:
                # Benchmark evaluation: all valid depth points in range
                valid_mask = (gt > args.eval_min_depth) & (gt < args.eval_max_depth)

            if not valid_mask.any():
                raise ValueError(
                    f"Frame {i} ({filenames[i]}) has zero valid ground truth pixels within "
                    f"[{args.eval_min_depth}, {args.eval_max_depth}]"
                )

            valid_gt = gt[valid_mask]
            valid_pred = pred_depth[valid_mask]

            # Median scaling: standard protocol for monocular depth estimation without scale supervision
            if not args.no_median_scaling:
                ratio = np.median(valid_gt) / np.median(valid_pred)
                valid_pred *= ratio

            # Clip predictions to evaluation depth limits
            valid_pred = np.clip(valid_pred, args.eval_min_depth, args.eval_max_depth)

            errors.append(compute_depth_errors(valid_gt, valid_pred))

            if (i + 1) % 100 == 0:
                print(f"Evaluated [{i+1}/{len(filenames)}]")

    assert len(errors) == len(filenames), f"Expected {len(filenames)} frames evaluated, got {len(errors)}"

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

    ref_str = "~0.090 (benchmark 652)" if is_benchmark else "~0.115 (raw 697)"
    print("\n" + "=" * 65)
    print(f" KITTI Eigen Results ({len(filenames)} frames, Target: {ref_str})")
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

    # Parity gate enforcement for official Monodepth2
    if args.model == "monodepth2_official":
        expected_abs_rel = 0.090 if is_benchmark else 0.115
        tol = args.gate_tolerance
        diff = abs(results["abs_rel"] - expected_abs_rel)
        if diff > tol:
            print("\n" + "!" * 70)
            print(f" [PARITY GATE FAILED] Official Monodepth2 AbsRel={results['abs_rel']:.4f} "
                  f"deviated from target {expected_abs_rel:.4f} by {diff:.4f} (tolerance: {tol:.4f})!")
            print(" Halting execution. Debug the evaluation pipeline before proceeding to training.")
            print("!" * 70 + "\n")
            sys.exit(1)
        else:
            print(f"\n[PARITY GATE PASSED] Official Monodepth2 AbsRel={results['abs_rel']:.4f} "
                  f"matches target {expected_abs_rel:.4f} within tolerance {tol:.4f}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XiDepth models on KITTI Eigen test benchmark")
    parser.add_argument("--model", type=str, default="xidepth", choices=["xidepth", "resnet18", "monodepth2", "monodepth2_official"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="data/kitti")
    parser.add_argument("--split_file", type=str, default="data/splits/eigen_benchmark/test_files.txt")
    parser.add_argument("--gt_path", type=str, default="data/gt_depths.npz")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--gate_tolerance", type=float, default=0.005, help="Parity gate tolerance for official model (default: 0.005)")

    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min_depth", type=float, default=0.1, help="Network inversion min depth")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Network inversion max depth")
    parser.add_argument("--eval_min_depth", type=float, default=1e-3, help="Benchmark evaluation min depth (default: 1e-3)")
    parser.add_argument("--eval_max_depth", type=float, default=80.0, help="Benchmark evaluation max depth (default: 80.0)")
    parser.add_argument("--no_median_scaling", action="store_true", default=False, help="Disable median scaling")
    parser.add_argument("--post_process", action="store_true", default=False, help="Enable flip post-processing (default: False)")
    parser.add_argument("--no_cuda", action="store_true", default=False)

    args = parser.parse_args()
    evaluate(args)
