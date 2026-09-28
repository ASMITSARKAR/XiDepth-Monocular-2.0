import torch
import pytest

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
