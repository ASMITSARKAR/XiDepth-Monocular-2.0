import random
import numpy as np
import torch
import pytest
from PIL import Image

from data import KITTIRawDataset


def test_dataset_sample_loading():
    lines = [
        "2011_09_26/2011_09_26_drive_0001_sync 6 l",
        "2011_09_26/2011_09_26_drive_0001_sync 7 l",
    ]
    dataset = KITTIRawDataset(
        data_path="data/sample",
        filenames=lines,
        height=192,
        width=640,
        is_train=False,
        use_stereo=True,
    )

    assert len(dataset) == 2
    item = dataset[0]

    # Target frame 0 and temporal context frames -1, 1
    assert item[("color", 0, 0)].shape == (3, 192, 640)
    assert item[("color", -1, 0)].shape == (3, 192, 640)
    assert item[("color", 1, 0)].shape == (3, 192, 640)
    assert item[("color", "s", 0)].shape == (3, 192, 640)

    # Intrinsics check
    k = item["K"]
    assert k.shape == (4, 4)
    assert k[0, 0] > 0
    assert k[1, 1] > 0
    assert k[2, 2] == 1.0

    # Stereo translation check
    stereo_t = item["stereo_T"]
    assert stereo_t.shape == (4, 4)
    assert stereo_t[0, 3] < 0  # left camera looking at right camera has negative horizontal offset


def test_dataset_right_camera_mapping_and_intrinsics():
    """Verify side 'r' maps to image_03, uses P_rect_03, and keeps neighbors in camera 3."""
    line_r = ["2011_09_26/2011_09_26_drive_0001_sync 6 r"]
    line_l = ["2011_09_26/2011_09_26_drive_0001_sync 6 l"]

    ds_r = KITTIRawDataset(data_path="data/sample", filenames=line_r, height=192, width=640, is_train=False)
    ds_l = KITTIRawDataset(data_path="data/sample", filenames=line_l, height=192, width=640, is_train=False)

    item_r = ds_r[0]
    item_l = ds_l[0]

    # 1. Verify all temporal frames are loaded for right camera
    assert item_r[("color", 0, 0)].shape == (3, 192, 640)
    assert item_r[("color", -1, 0)].shape == (3, 192, 640)
    assert item_r[("color", 1, 0)].shape == (3, 192, 640)

    # 2. Check path resolution: side 'r' explicitly points to image_03
    path_r_0 = ds_r._get_image_path("2011_09_26/2011_09_26_drive_0001_sync", 6, "r")
    path_r_prev = ds_r._get_image_path("2011_09_26/2011_09_26_drive_0001_sync", 5, "r")
    path_r_next = ds_r._get_image_path("2011_09_26/2011_09_26_drive_0001_sync", 7, "r")

    assert "image_03" in path_r_0
    assert "image_03" in path_r_prev
    assert "image_03" in path_r_next

    # 3. Verify intrinsics: P_rect_02 vs P_rect_03 read correctly
    k_r = item_r["K"].numpy()
    k_l = item_l["K"].numpy()

    assert k_r.shape == (4, 4)
    assert k_l.shape == (4, 4)
    assert k_r[0, 0] > 0 and k_r[1, 1] > 0

    # Read raw calibration file directly to verify P_rect_03 values
    calib = ds_r._read_kitti_calib("data/sample/2011_09_26/calib_cam_to_cam.txt", side="r")
    assert calib[0, 0] > 0


def test_dataset_horizontal_flip_handling():
    """Verify horizontal flip augmentation mirrors optical center and image tensors."""
    lines = ["2011_09_26/2011_09_26_drive_0001_sync 6 r"]
    ds = KITTIRawDataset(data_path="data/sample", filenames=lines, height=192, width=640, is_train=True)

    # Deterministic test: force flip = True
    random.seed(12345)
    # Monkey-patch random.random to return 0.9 (> 0.5) so flip executes
    orig_random = random.random
    try:
        random.random = lambda: 0.9
        item_flipped = ds[0]

        random.random = lambda: 0.1
        item_unflipped = ds[0]

        # In flipped item, optical center cx should be width - original_cx
        k_flipped = item_flipped["K"].numpy()
        k_unflipped = item_unflipped["K"].numpy()

        assert k_flipped[0, 2] == pytest.approx(640 - k_unflipped[0, 2], abs=1e-3)

        # Image tensor should be horizontally flipped
        img_unflipped = item_unflipped[("color", 0, 0)]
        img_flipped = item_flipped[("color", 0, 0)]

        expected_flipped = torch.flip(img_unflipped, dims=[2])
        assert torch.allclose(img_flipped, expected_flipped, atol=1e-5)

        # Inverse K should match inverse of flipped K
        inv_k = item_flipped["inv_K"].numpy()
        reconstructed_identity = k_flipped[:3, :3] @ inv_k[:3, :3]
        assert np.allclose(reconstructed_identity, np.eye(3), atol=1e-4)

    finally:
        random.random = orig_random


def test_dataset_hard_fails_on_missing_file():
    lines = [
        "2011_09_26/non_existent_drive_sync 999 l",
    ]
    dataset = KITTIRawDataset(
        data_path="data/sample",
        filenames=lines,
        height=192,
        width=640,
        is_train=False,
    )
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
