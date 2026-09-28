from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.xi_block import XiBlock


def disp_to_depth(
    disp: torch.Tensor,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_refl: bool = True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.conv(self.pad(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class XiDepthNet(nn.Module):
    def __init__(self, num_scales: int = 4, bias_init: float = -4.5):
        super().__init__()
        self.num_scales = num_scales

        # Backbone stages
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(24, 48, 3)    # 1/4
        self.stage3 = self._make_stage(48, 96, 3)    # 1/8
        self.stage4 = self._make_stage(96, 192, 3)   # 1/16
        self.stage5 = self._make_stage(192, 384, 3)  # 1/32

        # Decoder stages
        self.upconv5 = ConvBlock(384, 192)
        self.upconv4 = ConvBlock(192 + 192, 96)
        self.upconv3 = ConvBlock(96 + 96, 48)
        self.upconv2 = ConvBlock(48 + 48, 24)
        self.upconv1 = ConvBlock(24 + 24, 16)

        self.iconv5 = Conv3x3(192, 192)
        self.iconv4 = Conv3x3(96, 96)
        self.iconv3 = Conv3x3(48, 48)
        self.iconv2 = Conv3x3(24, 24)
        self.iconv1 = Conv3x3(16, 16)

        self.disp4 = nn.Sequential(nn.Conv2d(96, 1, 3, padding=1), nn.Sigmoid())
        self.disp3 = nn.Sequential(nn.Conv2d(48, 1, 3, padding=1), nn.Sigmoid())
        self.disp2 = nn.Sequential(nn.Conv2d(24, 1, 3, padding=1), nn.Sigmoid())
        self.disp1 = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

        # Disparity conv bias initialization to avoid near-plane collapse
        for disp_head in [self.disp1, self.disp2, self.disp3, self.disp4]:
            nn.init.constant_(disp_head[0].bias, bias_init)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [XiBlock(in_channels, out_channels, stride=2)]
        for _ in range(1, num_blocks):
            layers.append(XiBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        conv1 = self.conv1(x)
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)
        conv5 = self.stage5(conv4)

        up5 = F.interpolate(conv5, scale_factor=2, mode="nearest")
        iconv5 = self.iconv5(self.upconv5(up5))

        concat4 = torch.cat([iconv5, conv4], dim=1)
        up4 = F.interpolate(concat4, scale_factor=2, mode="nearest")
        iconv4 = self.iconv4(self.upconv4(up4))
        disp4 = self.disp4(iconv4)

        concat3 = torch.cat([iconv4, conv3], dim=1)
        up3 = F.interpolate(concat3, scale_factor=2, mode="nearest")
        iconv3 = self.iconv3(self.upconv3(up3))
        disp3 = self.disp3(iconv3)

        concat2 = torch.cat([iconv3, conv2], dim=1)
        up2 = F.interpolate(concat2, scale_factor=2, mode="nearest")
        iconv2 = self.iconv2(self.upconv2(up2))
        disp2 = self.disp2(iconv2)

        concat1 = torch.cat([iconv2, conv1], dim=1)
        up1 = F.interpolate(concat1, scale_factor=2, mode="nearest")
        iconv1 = self.iconv1(self.upconv1(up1))
        disp1 = self.disp1(iconv1)

        outputs = [disp1, disp2, disp3, disp4]
        if self.training:
            return outputs[: self.num_scales]
        return outputs[0]
