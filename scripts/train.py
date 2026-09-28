import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig, DatasetConfig
from data.kitti_raw_dataset import KITTIRawDataset
from models import XiDepthNet, ResNetDepthNet, PoseNet, disp_to_depth
from utils.geometry import BackprojectDepth, Project3D, transformation_from_parameters
from utils.loss import compute_reprojection_loss, compute_smoothness_loss
from utils.health import DisparityHealthMonitor


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(f"Training on device: {self.device}")

        # Model setup
        if args.model == "xidepth":
            self.depth_net = XiDepthNet(num_scales=args.num_scales, bias_init=args.disp_bias_init)
        elif args.model == "resnet18":
            self.depth_net = ResNetDepthNet(
                num_scales=args.num_scales,
                pretrained=args.pretrained,
                bias_init=args.disp_bias_init,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        self.pose_net = PoseNet(num_input_images=2, pretrained=args.pretrained)

        self.depth_net.to(self.device)
        self.pose_net.to(self.device)

        # Geometry modules
        self.backproject = BackprojectDepth(args.height, args.width).to(self.device)
        self.project = Project3D(args.height, args.width).to(self.device)

        # Optimizer and scheduler
        parameters = list(self.depth_net.parameters()) + list(self.pose_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and self.device.type == "cuda"))
        self.health_monitor = DisparityHealthMonitor(
            min_std=args.min_disp_std,
            max_consecutive_low_variance=args.max_consecutive_low_variance,
        )

        self._setup_datasets()

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")

    def _setup_datasets(self):
        train_file = os.path.join(self.args.split_dir, "train_files.txt")
        val_file = os.path.join(self.args.split_dir, "val_files.txt")

        with open(train_file, "r") as f:
            train_filenames = [l.strip() for l in f if l.strip()]

        with open(val_file, "r") as f:
            val_filenames = [l.strip() for l in f if l.strip()]

        if self.args.max_samples > 0:
            train_filenames = train_filenames[: self.args.max_samples]
            val_filenames = val_filenames[: max(1, self.args.max_samples // 5)]

        print(f"Dataset split: {len(train_filenames)} train frames, {len(val_filenames)} val frames")

        self.train_dataset = KITTIRawDataset(
            data_path=self.args.dataset_dir,
            filenames=train_filenames,
            height=self.args.height,
            width=self.args.width,
            frame_ids=self.args.frame_ids,
            is_train=True,
            use_stereo=self.args.use_stereo,
        )

        self.val_dataset = KITTIRawDataset(
            data_path=self.args.dataset_dir,
            filenames=val_filenames,
            height=self.args.height,
            width=self.args.width,
            frame_ids=self.args.frame_ids,
            is_train=False,
            use_stereo=self.args.use_stereo,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _compute_reprojection(
        self,
        depth: torch.Tensor,
        inv_k: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
        source_img: torch.Tensor,
    ) -> torch.Tensor:
        cam_points = self.backproject(depth, inv_k)
        pix_coords = self.project(cam_points, k, t)
        return F.grid_sample(source_img, pix_coords, mode="bilinear", padding_mode="border", align_corners=True)

    def _compute_losses(self, inputs: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        target_img = inputs[("color", 0, 0)].to(self.device)
        target_aug = inputs[("color_aug", 0, 0)].to(self.device)
        k = inputs["K"].to(self.device)
        inv_k = inputs["inv_K"].to(self.device)

        disps = self.depth_net(target_aug)
        if not isinstance(disps, list):
            disps = [disps]

        # Health monitor on primary full-resolution disparity
        disp_metrics = self.health_monitor.check(disps[0])

        # Estimate camera ego-motion relative to temporal source frames
        poses = {}
        for f_id in [fid for fid in self.args.frame_ids if fid != 0]:
            source_aug = inputs[("color_aug", f_id, 0)].to(self.device)
            if f_id < 0:
                pose_inputs = torch.cat([source_aug, target_aug], dim=1)
                axisangle, translation = self.pose_net(pose_inputs)
                poses[f_id] = transformation_from_parameters(axisangle, translation, invert=True)
            else:
                pose_inputs = torch.cat([target_aug, source_aug], dim=1)
                axisangle, translation = self.pose_net(pose_inputs)
                poses[f_id] = transformation_from_parameters(axisangle, translation, invert=False)

        # Include stereo transform if active
        if self.args.use_stereo and "stereo_T" in inputs:
            poses["s"] = inputs["stereo_T"].to(self.device)

        total_loss = 0.0
        loss_dict: Dict[str, float] = {}

        source_frames = [fid for fid in self.args.frame_ids if fid != 0]
        if self.args.use_stereo and "stereo_T" in inputs:
            source_frames.append("s")

        # Identity reprojection error for auto-masking (moving objects / static scenes)
        identity_reproj_losses = []
        for f_id in source_frames:
            source_raw = inputs[("color", f_id, 0)].to(self.device)
            identity_reproj_losses.append(compute_reprojection_loss(source_raw, target_img))
        identity_reproj_loss = torch.cat(identity_reproj_losses, dim=1)

        # Multi-scale minimum reprojection + smoothness
        for s, disp in enumerate(disps):
            # Upsample disparity to full image resolution to prevent texture copying artifacts
            disp_up = F.interpolate(
                disp, size=(self.args.height, self.args.width), mode="bilinear", align_corners=False
            )
            _, depth_up = disp_to_depth(disp_up, self.args.min_depth, self.args.max_depth)

            warped_reproj_losses = []
            for f_id in source_frames:
                source_raw = inputs[("color", f_id, 0)].to(self.device)
                warped_img = self._compute_reprojection(depth_up, inv_k, k, poses[f_id], source_raw)
                warped_reproj_losses.append(compute_reprojection_loss(warped_img, target_img))

            warped_reproj_loss = torch.cat(warped_reproj_losses, dim=1)

            # Minimum reprojection loss over all available views
            min_reproj_loss, _ = torch.min(warped_reproj_loss, dim=1, keepdim=True)

            # Auto-masking: ignore pixels where unwarped source is closer than warped projection
            if self.args.use_automask:
                min_identity_loss, _ = torch.min(identity_reproj_loss, dim=1, keepdim=True)
                auto_mask = (min_reproj_loss < min_identity_loss).float()
                photo_loss = (min_reproj_loss * auto_mask).sum() / (auto_mask.sum() + 1e-7)
            else:
                photo_loss = min_reproj_loss.mean()

            # Edge-aware smoothness
            smooth_loss = compute_smoothness_loss(disp, target_img)
            scale_loss = photo_loss + self.args.smoothness_weight * smooth_loss / (2 ** s)

            total_loss += scale_loss
            loss_dict[f"photo_loss_scale_{s}"] = photo_loss.item()
            loss_dict[f"smooth_loss_scale_{s}"] = smooth_loss.item()

        total_loss /= len(disps)
        loss_dict["total_loss"] = total_loss.item()
        loss_dict.update(disp_metrics)

        return total_loss, loss_dict

    def train_epoch(self, epoch: int) -> float:
        self.depth_net.train()
        self.pose_net.train()
        epoch_losses = []

        start_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(self.args.use_amp and self.device.type == "cuda")):
                loss, loss_dict = self._compute_losses(inputs)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % self.args.log_freq == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{self.args.epochs}] Step [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} | Disp std: {loss_dict['disp_std']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return sum(epoch_losses) / len(epoch_losses)

    def validate(self) -> float:
        self.depth_net.eval()
        self.pose_net.eval()
        val_losses = []

        with torch.no_grad():
            for inputs in self.val_loader:
                with torch.amp.autocast("cuda", enabled=(self.args.use_amp and self.device.type == "cuda")):
                    loss, _ = self._compute_losses(inputs)
                val_losses.append(loss.item())

        return sum(val_losses) / len(val_losses) if val_losses else float("inf")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "depth_net": self.depth_net.state_dict(),
            "pose_net": self.pose_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": vars(self.args),
        }
        ckpt_path = os.path.join(self.args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(state, ckpt_path)

        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"Saved new best model checkpoint to {best_path}")

    def run(self):
        print(f"Beginning training {self.args.model} for {self.args.epochs} epochs...")
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch+1} Complete | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)


def get_args():
    parser = argparse.ArgumentParser(description="XiDepth v2.0 Training Pipeline")
    parser.add_argument("--model", type=str, default="xidepth", choices=["xidepth", "resnet18"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--disp_bias_init", type=float, default=-4.5)

    parser.add_argument("--dataset_dir", type=str, default="data/sample")
    parser.add_argument("--split_dir", type=str, default="data/splits/eigen_zhou")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--frame_ids", nargs="+", type=int, default=[0, -1, 1])
    parser.add_argument("--use_stereo", action="store_true", default=False)
    parser.add_argument("--use_automask", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=15)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--smoothness_weight", type=float, default=0.001)

    parser.add_argument("--min_disp_std", type=float, default=0.005)
    parser.add_argument("--max_consecutive_low_variance", type=int, default=5)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args)
    trainer.run()
