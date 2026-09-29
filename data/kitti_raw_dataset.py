import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class KITTIRawDataset(Dataset):
    DEFAULT_K = np.array([
        [721.5377, 0.0, 609.5593, 0.0],
        [0.0, 721.5377, 172.8540, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    STEREO_BASELINE = 0.54

    def __init__(
        self,
        data_path: str,
        filenames: List[str],
        height: int = 192,
        width: int = 640,
        frame_ids: Optional[List[int]] = None,
        is_train: bool = True,
        use_stereo: bool = False,
    ):
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_ids = frame_ids if frame_ids is not None else [0, -1, 1]
        self.is_train = is_train
        self.use_stereo = use_stereo

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self._calib_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _parse_line(self, line: str) -> Tuple[str, int, str]:
        parts = line.strip().split()
        folder = parts[0]
        frame_idx = int(parts[1])
        side = parts[2] if len(parts) > 2 else "l"
        return folder, frame_idx, side

    def _get_image_path(self, folder: str, frame_idx: int, side: str) -> str:
        cam_dir = "image_02" if side == "l" else "image_03"
        frame_name = f"{frame_idx:010d}"

        # Search candidates: full raw tree, direct folder, or flat structure
        candidates = [
            os.path.join(self.data_path, folder, cam_dir, "data", f"{frame_name}.png"),
            os.path.join(self.data_path, folder, cam_dir, "data", f"{frame_name}.jpg"),
            os.path.join(self.data_path, folder, cam_dir, f"{frame_name}.png"),
            os.path.join(self.data_path, folder, cam_dir, f"{frame_name}.jpg"),
            os.path.join(self.data_path, folder, f"{frame_name}.png"),
            os.path.join(self.data_path, folder, f"{frame_name}.jpg"),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f"Frame {frame_name} in folder {folder} (side {side}, camera {cam_dir}) not found under {self.data_path}"
        )

    def _load_image(self, path: str) -> Image.Image:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found at {path}")
        return Image.open(path).convert("RGB")

    def _get_intrinsics(self, folder: str, orig_w: int, orig_h: int, side: str = "l") -> np.ndarray:
        date = folder.split("/")[0] if "/" in folder else folder.split("\\")[0]
        cache_key = f"{date}_{side}"
        if cache_key in self._calib_cache:
            k = self._calib_cache[cache_key].copy()
        else:
            calib_file = os.path.join(self.data_path, date, "calib_cam_to_cam.txt")
            k = self._read_kitti_calib(calib_file, side=side)
            self._calib_cache[cache_key] = k.copy()

        # Scale K to target network resolution
        k[0, :] *= self.width / orig_w
        k[1, :] *= self.height / orig_h
        return k

    def _read_kitti_calib(self, calib_file: str, side: str = "l") -> np.ndarray:
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"Calibration file missing: {calib_file}")

        prefix = "P_rect_02:" if side == "l" else "P_rect_03:"
        alt_prefix = "P2:" if side == "l" else "P3:"

        with open(calib_file, "r") as f:
            for line in f:
                if line.startswith(prefix) or line.startswith(alt_prefix):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    k = np.eye(4, dtype=np.float32)
                    k[:3, :3] = np.array(vals[:12], dtype=np.float32).reshape(3, 4)[:3, :3]
                    return k

        raise ValueError(f"Could not find projection matrix for camera {side} ({prefix}) in {calib_file}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        folder, frame_idx, side = self._parse_line(self.filenames[idx])

        # Load reference frame (id = 0)
        img_0 = self._load_image(self._get_image_path(folder, frame_idx, side))
        orig_w, orig_h = img_0.size

        # Intrinsic matrix for specified camera side
        k = self._get_intrinsics(folder, orig_w, orig_h, side=side)


        # Horizontal flip augmentation
        do_flip = self.is_train and random.random() > 0.5
        if do_flip:
            # Mirror optical center horizontal coordinate
            k[0, 2] = self.width - k[0, 2]

        # Color jitter params
        do_color_aug = self.is_train and random.random() > 0.5
        b_factor = random.uniform(*self.brightness) if do_color_aug else 1.0
        c_factor = random.uniform(*self.contrast) if do_color_aug else 1.0
        s_factor = random.uniform(*self.saturation) if do_color_aug else 1.0
        h_factor = random.uniform(*self.hue) if do_color_aug else 0.0

        inputs: Dict[str, torch.Tensor] = {}

        # Load temporal frames
        for f_id in self.frame_ids:
            target_idx = frame_idx + f_id
            img = self._load_image(self._get_image_path(folder, target_idx, side))

            # Resize to target network dimensions
            img_resized = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
            if do_flip:
                img_resized = TF.hflip(img_resized)

            # Raw normalized tensor
            inputs[("color", f_id, 0)] = TF.to_tensor(img_resized)

            # Augmented tensor for photometric robustness
            if do_color_aug:
                img_aug = TF.adjust_brightness(img_resized, b_factor)
                img_aug = TF.adjust_contrast(img_aug, c_factor)
                img_aug = TF.adjust_saturation(img_aug, s_factor)
                img_aug = TF.adjust_hue(img_aug, h_factor)
                inputs[("color_aug", f_id, 0)] = TF.to_tensor(img_aug)
            else:
                inputs[("color_aug", f_id, 0)] = inputs[("color", f_id, 0)]

        # Stereo right camera frame if requested
        if self.use_stereo:
            stereo_side = "r" if side == "l" else "l"
            s_img = self._load_image(self._get_image_path(folder, frame_idx, stereo_side))
            s_resized = s_img.resize((self.width, self.height), Image.Resampling.BILINEAR)
            if do_flip:
                s_resized = TF.hflip(s_resized)

            inputs[("color", "s", 0)] = TF.to_tensor(s_resized)
            if do_color_aug:
                s_aug = TF.adjust_brightness(s_resized, b_factor)
                s_aug = TF.adjust_contrast(s_aug, c_factor)
                s_aug = TF.adjust_saturation(s_aug, s_factor)
                s_aug = TF.adjust_hue(s_aug, h_factor)
                inputs[("color_aug", "s", 0)] = TF.to_tensor(s_aug)
            else:
                inputs[("color_aug", "s", 0)] = inputs[("color", "s", 0)]

            # Stereo baseline transform
            stereo_t = np.eye(4, dtype=np.float32)
            baseline_sign = -1.0 if side == "l" else 1.0
            if do_flip:
                baseline_sign *= -1.0
            stereo_t[0, 3] = baseline_sign * self.STEREO_BASELINE
            inputs["stereo_T"] = torch.from_numpy(stereo_t)

        inv_k = np.linalg.pinv(k[:3, :3])
        inv_k_4x4 = np.eye(4, dtype=np.float32)
        inv_k_4x4[:3, :3] = inv_k

        inputs["K"] = torch.from_numpy(k)
        inputs["inv_K"] = torch.from_numpy(inv_k_4x4)

        return inputs
