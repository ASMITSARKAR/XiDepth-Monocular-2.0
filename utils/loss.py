import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = torch.clamp(self.sig_x_pool(x ** 2) - mu_x ** 2, min=0.0)
        sigma_y = torch.clamp(self.sig_y_pool(y ** 2) - mu_y ** 2, min=0.0)
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        ssim_n = (2.0 * mu_x * mu_y + self.c1) * (2.0 * sigma_xy + self.c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)

        return torch.clamp((1.0 - ssim_n / ssim_d) / 2.0, 0.0, 1.0)


_ssim_module = SSIM()


def compute_reprojection_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, keepdim=True)
    ssim_loss = _ssim_module(pred, target).mean(1, keepdim=True)
    return 0.85 * ssim_loss + 0.15 * l1_loss


def compute_smoothness_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] != disp.shape[-2:]:
        img = F.interpolate(img, size=disp.shape[-2:], mode="area")

    mean_disp = disp.mean(dim=(2, 3), keepdim=True)
    norm_disp = disp / (mean_disp + 1e-7)

    disp_dx = norm_disp[:, :, :, 1:] - norm_disp[:, :, :, :-1]
    disp_dy = norm_disp[:, :, 1:, :] - norm_disp[:, :, :-1, :]

    image_dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    image_dy = img[:, :, 1:, :] - img[:, :, :-1, :]

    weight_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = disp_dx * weight_x
    smoothness_y = disp_dy * weight_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()
