from typing import Dict
import torch


class DisparityHealthMonitor:
    def __init__(
        self,
        min_std: float = 0.005,
        max_consecutive_low_variance: int = 5,
    ):
        self.min_std = min_std
        self.max_consecutive = max_consecutive_low_variance
        self.consecutive_low_variance = 0

    def check(self, disp: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            disp_detached = disp.detach()

            if torch.isnan(disp_detached).any() or torch.isinf(disp_detached).any():
                raise RuntimeError("Numerical instability detected: NaN or Inf in disparity tensor.")

            mean_val = disp_detached.mean().item()
            std_val = disp_detached.std().item()
            min_val = disp_detached.min().item()
            max_val = disp_detached.max().item()

            if std_val < self.min_std:
                self.consecutive_low_variance += 1
                if self.consecutive_low_variance >= self.max_consecutive:
                    raise RuntimeError(
                        f"Disparity collapse detected! Spatial std has been {std_val:.6f} "
                        f"(< {self.min_std}) for {self.consecutive_low_variance} consecutive batches."
                    )
            else:
                self.consecutive_low_variance = 0

            return {
                "disp_mean": mean_val,
                "disp_std": std_val,
                "disp_min": min_val,
                "disp_max": max_val,
            }
