import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def find_image(dataset_dir: str, folder: str, cam_dir: str, frame_idx: int) -> bool:
    frame_name = f"{frame_idx:010d}"
    candidates = [
        os.path.join(dataset_dir, folder, cam_dir, "data", f"{frame_name}.png"),
        os.path.join(dataset_dir, folder, cam_dir, "data", f"{frame_name}.jpg"),
        os.path.join(dataset_dir, folder, cam_dir, f"{frame_name}.png"),
        os.path.join(dataset_dir, folder, cam_dir, f"{frame_name}.jpg"),
        os.path.join(dataset_dir, folder, f"{frame_name}.png"),
        os.path.join(dataset_dir, folder, f"{frame_name}.jpg"),
    ]
    return any(os.path.isfile(p) for p in candidates)


def verify_split(dataset_dir: str, split_file: str, is_train: bool = True) -> int:
    if not os.path.isfile(split_file):
        print(f"  [SKIPPED] Split file not found: {split_file}")
        return 0

    with open(split_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\nVerifying {len(lines)} entries in {split_file} (temporal checks={is_train})...")
    missing_count = 0
    missing_samples = []

    # Check calibration files for each unique date
    dates = set(l.split()[0].split("/")[0] for l in lines)
    for date in dates:
        calib_file = os.path.join(dataset_dir, date, "calib_cam_to_cam.txt")
        if not os.path.isfile(calib_file):
            print(f"  [MISSING CALIB] {calib_file}")
            missing_count += 1

    # Check images
    frame_offsets = [0, -1, 1] if is_train else [0]
    for idx, line in enumerate(lines):
        parts = line.split()
        folder = parts[0]
        frame_idx = int(parts[1])
        side = parts[2] if len(parts) > 2 else "l"
        cam_dir = "image_02" if side == "l" else "image_03"

        for offset in frame_offsets:
            target_frame = frame_idx + offset
            if not find_image(dataset_dir, folder, cam_dir, target_frame):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(f"{folder}/{cam_dir}/{target_frame:010d}")

        if (idx + 1) % 5000 == 0:
            print(f"  Checked [{idx+1}/{len(lines)}] entries...")

    if missing_count > 0:
        print(f"\n[FAIL] Found {missing_count} missing files in {split_file}!")
        print("First missing samples:", missing_samples)
    else:
        print(f"[PASS] All files for {split_file} ({len(lines)} entries) verified successfully.")

    return missing_count


def main():
    parser = argparse.ArgumentParser(description="Verify all dataset paths in KITTI splits exist on disk")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to KITTI dataset root")
    parser.add_argument("--split_dir", type=str, default="data/splits", help="Path to splits base dir")
    args = parser.parse_args()

    train_file = os.path.join(args.split_dir, "eigen_zhou", "train_files.txt")
    val_file = os.path.join(args.split_dir, "eigen_zhou", "val_files.txt")
    test_bench = os.path.join(args.split_dir, "eigen_benchmark", "test_files.txt")
    test_raw = os.path.join(args.split_dir, "eigen", "test_files.txt")

    total_missing = 0
    total_missing += verify_split(args.dataset_dir, train_file, is_train=True)
    total_missing += verify_split(args.dataset_dir, val_file, is_train=True)
    total_missing += verify_split(args.dataset_dir, test_bench, is_train=False)
    total_missing += verify_split(args.dataset_dir, test_raw, is_train=False)

    print("\n" + "=" * 65)
    if total_missing == 0:
        print(" DATASET INTEGRITY VERIFIED: ALL PATHS & CALIBRATION EXIST")
    else:
        print(f" DATASET INTEGRITY FAILED: {total_missing} MISSING FILES DETECTED")
    print("=" * 65)

    if total_missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
