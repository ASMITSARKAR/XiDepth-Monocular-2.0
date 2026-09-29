from typing import Dict
import torch

from models.depth_net import disp_to_depth


class DisparityHealthMonitor:
    """Watchdog and telemetry monitor for self-supervised monocular depth estimation.
    
    Monitors:
    1. Numerical Stability: Traps NaN / Inf immediately before corrupted gradients poison weights.
    2. Physical Depth Calibration: Tracks metric depth distribution (mean, std, min, max in meters).
    3. Degenerate Mode Collapse Detection: Detects true collapse (flat dead outputs where
       spatial std < 1e-6 or depth saturated to near/far plane boundaries) over sustained periods.
    """

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        warmup_steps: int = 1000,
        dead_neuron_std_thresh: float = 1e-6,
        max_consecutive_collapse: int = 100,
        fail_on_collapse: bool = False,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.warmup_steps = warmup_steps
        self.dead_neuron_std_thresh = dead_neuron_std_thresh
        self.max_consecutive_collapse = max_consecutive_collapse
        self.fail_on_collapse = fail_on_collapse

        self.step_count = 0
        self.consecutive_collapse_batches = 0

    def check(self, disp: torch.Tensor) -> Dict[str, float]:
        """Inspect a batch of predicted disparities (shape [B, 1, H, W], sigmoid output in (0, 1)).
        
        Returns a dictionary of disparity and physical depth telemetry metrics.
        """
        with torch.no_grad():
            self.step_count += 1
            disp_detached = disp.detach()

            # 1. Immediate numerical stability check
            if torch.isnan(disp_detached).any() or torch.isinf(disp_detached).any():
                raise RuntimeError(
                    f"[Health Critical] Numerical instability detected at step {self.step_count}: "
                    "Disparity tensor contains NaN or Inf values."
                )

            disp_mean = disp_detached.mean().item()
            disp_std = disp_detached.std().item()
            disp_min = disp_detached.min().item()
            disp_max = disp_detached.max().item()

            # 2. Physical metric depth computation
            _, depth = disp_to_depth(disp_detached, self.min_depth, self.max_depth)
            depth_mean = depth.mean().item()
            depth_std = depth.std().item()
            depth_min = depth.min().item()
            depth_max = depth.max().item()

            # 3. Collapse detection after warmup period
            is_collapsed = False
            collapse_reason = ""

            if self.step_count > self.warmup_steps:
                if disp_std < self.dead_neuron_std_thresh:
                    is_collapsed = True
                    collapse_reason = (
                        f"Zero spatial variance (disp_std={disp_std:.2e} < {self.dead_neuron_std_thresh:.2e})"
                    )
                elif depth_mean < (self.min_depth + 0.2):
                    is_collapsed = True
                    collapse_reason = f"Near-plane saturation (depth_mean={depth_mean:.2f}m)"
                elif depth_mean > (self.max_depth - 5.0):
                    is_collapsed = True
                    collapse_reason = f"Far-plane saturation (depth_mean={depth_mean:.2f}m)"

                if is_collapsed:
                    self.consecutive_collapse_batches += 1
                    if self.consecutive_collapse_batches >= self.max_consecutive_collapse:
                        msg = (
                            f"[Health Warning] Sustained collapse detected: {collapse_reason} "
                            f"for {self.consecutive_collapse_batches} consecutive batches at step {self.step_count}."
                        )
                        if self.fail_on_collapse:
                            raise RuntimeError(msg)
                        elif (
                            self.consecutive_collapse_batches == self.max_consecutive_collapse
                            or self.consecutive_collapse_batches % 100 == 0
                        ):
                            print(msg)
                else:
                    self.consecutive_collapse_batches = 0

            return {
                "disp_mean": disp_mean,
                "disp_std": disp_std,
                "disp_min": disp_min,
                "disp_max": disp_max,
                "depth_mean": depth_mean,
                "depth_std": depth_std,
                "depth_min": depth_min,
                "depth_max": depth_max,
            }

    def state_dict(self) -> Dict[str, int]:
        return {
            "step_count": self.step_count,
            "consecutive_collapse_batches": self.consecutive_collapse_batches,
        }

    def load_state_dict(self, state: Dict[str, int]):
        self.step_count = state.get("step_count", 0)
        self.consecutive_collapse_batches = state.get("consecutive_collapse_batches", 0)
