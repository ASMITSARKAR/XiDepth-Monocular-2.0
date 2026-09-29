import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def read_calib_file(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file missing: {path}")
    data = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.strip().split()])
    return data


def generate_depth_map(calib_dir: str, velo_filename: str, cam: int = 2) -> np.ndarray:
    if not os.path.isfile(velo_filename):
        raise FileNotFoundError(f"Velodyne scan file missing: {velo_filename}")

    cam_to_cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    velo_to_cam = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))

    # Extrinsics: Velodyne to camera
    R_velo_to_cam = velo_to_cam["R"].reshape(3, 3)
    T_velo_to_cam = velo_to_cam["T"]
    RT_velo_to_cam = np.eye(4)
    RT_velo_to_cam[:3, :3] = R_velo_to_cam
    RT_velo_to_cam[:3, 3] = T_velo_to_cam

    # Camera rectifying rotation
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam_to_cam["R_rect_00"].reshape(3, 3)

    # Projection matrix
    P_rect = cam_to_cam[f"P_rect_{cam:02d}"].reshape(3, 4)

    # Load velodyne points
    scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform to camera coordinate frame
    points_cam = (R_rect @ RT_velo_to_cam @ points_homo.T).T
    # Keep points in front of camera
    points_cam = points_cam[points_cam[:, 2] > 0]

    # Project to 2D
    points_2d = (P_rect @ points_cam.T).T
    points_2d[:, :2] /= points_2d[:, 2:3]

    # Assume standard KITTI resolution
    im_shape = (375, 1242)
    depth = np.zeros(im_shape, dtype=np.float32)

    x = np.round(points_2d[:, 0]).astype(int)
    y = np.round(points_2d[:, 1]).astype(int)
    val = points_cam[:, 2]

    valid = (x >= 0) & (x < im_shape[1]) & (y >= 0) & (y < im_shape[0])
    x, y, val = x[valid], y[valid], val[valid]

    # Sort descending by depth so closer points overwrite farther points
    sort_idx = np.argsort(val)[::-1]
    depth[y[sort_idx], x[sort_idx]] = val[sort_idx]

    return depth


def export_eigen_gt(kitti_dir: str, split_file: str, output_path: str):
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file missing: {split_file}")

    with open(split_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Exporting ground truth depth maps for {len(lines)} test frames from {split_file}...")
    gt_depths = []

    for i, line in enumerate(lines):
        parts = line.split()
        folder = parts[0]
        frame_idx = int(parts[1])
        date = folder.split("/")[0]

        velo_file = os.path.join(
            kitti_dir, folder, "velodyne_points", "data", f"{frame_idx:010d}.bin"
        )
        calib_dir = os.path.join(kitti_dir, date)

        if not os.path.isfile(velo_file):
            raise FileNotFoundError(f"Velodyne file missing for frame {i} ({folder} frame {frame_idx}): {velo_file}")
        if not os.path.isdir(calib_dir):
            raise FileNotFoundError(f"Calibration directory missing for frame {i} ({date}): {calib_dir}")

        depth = generate_depth_map(calib_dir, velo_file)
        gt_depths.append(depth)

        if (i + 1) % 100 == 0:
            print(f"Processed [{i+1}/{len(lines)}] frames")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, data=np.array(gt_depths, dtype=object))
    print(f"Successfully saved {len(gt_depths)} ground truth depth maps to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KITTI Velodyne LiDAR depth maps for Eigen test split")
    parser.add_argument("--kitti_dir", type=str, default="data/kitti")
    parser.add_argument("--split_file", type=str, default="data/splits/eigen/test_files.txt", help="Default: eigen 697 test split")
    parser.add_argument("--output_path", type=str, default="data/gt_depths_raw697.npz", help="Output .npz path")
    args = parser.parse_args()

    export_eigen_gt(args.kitti_dir, args.split_file, args.output_path)
