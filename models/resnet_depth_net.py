from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from models.depth_net import Conv3x3, ConvBlock


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, 1/4 resolution (with maxpool)
        self.layer2 = resnet.layer2  # 128 channels, 1/8 resolution
        self.layer3 = resnet.layer3  # 256 channels, 1/16 resolution
        self.layer4 = resnet.layer4  # 512 channels, 1/32 resolution

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        features.append(self.relu(x))  # 64 ch, 1/2 resolution
        features.append(self.layer1(self.maxpool(features[-1])))  # 64 ch, 1/4 resolution
        features.append(self.layer2(features[-1]))  # 128 ch, 1/8 resolution
        features.append(self.layer3(features[-1]))  # 256 ch, 1/16 resolution
        features.append(self.layer4(features[-1]))  # 512 ch, 1/32 resolution
        return features


class DepthDecoder(nn.Module):
    def __init__(self, num_scales: int = 4, bias_init: float = -4.5):
        super().__init__()
        self.num_scales = num_scales

        self.upconv5 = ConvBlock(512, 256)
        self.iconv5 = Conv3x3(256, 256)

        self.upconv4 = ConvBlock(256 + 256, 128)
        self.iconv4 = Conv3x3(128, 128)
        self.disp4 = nn.Sequential(nn.Conv2d(128, 1, 3, padding=1), nn.Sigmoid())

        self.upconv3 = ConvBlock(128 + 128, 64)
        self.iconv3 = Conv3x3(64, 64)
        self.disp3 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

        self.upconv2 = ConvBlock(64 + 64, 32)
        self.iconv2 = Conv3x3(32, 32)
        self.disp2 = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

        self.upconv1 = ConvBlock(32 + 64, 16)
        self.iconv1 = Conv3x3(16, 16)

        self.upconv0 = ConvBlock(16, 16)
        self.iconv0 = Conv3x3(16, 16)
        self.disp1 = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

        for disp_head in [self.disp1, self.disp2, self.disp3, self.disp4]:
            nn.init.constant_(disp_head[0].bias, bias_init)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        f0, f1, f2, f3, f4 = features

        up5 = F.interpolate(f4, scale_factor=2, mode="nearest")
        iconv5 = self.iconv5(self.upconv5(up5))

        concat4 = torch.cat([iconv5, f3], dim=1)
        up4 = F.interpolate(concat4, scale_factor=2, mode="nearest")
        iconv4 = self.iconv4(self.upconv4(up4))
        disp4 = self.disp4(iconv4)  # 1/8

        concat3 = torch.cat([iconv4, f2], dim=1)
        up3 = F.interpolate(concat3, scale_factor=2, mode="nearest")
        iconv3 = self.iconv3(self.upconv3(up3))
        disp3 = self.disp3(iconv3)  # 1/4

        concat2 = torch.cat([iconv3, f1], dim=1)
        up2 = F.interpolate(concat2, scale_factor=2, mode="nearest")
        iconv2 = self.iconv2(self.upconv2(up2))
        disp2 = self.disp2(iconv2)  # 1/2

        concat1 = torch.cat([iconv2, f0], dim=1)
        up1 = F.interpolate(concat1, scale_factor=2, mode="nearest")
        iconv1 = self.iconv1(self.upconv1(up1))

        up0 = F.interpolate(iconv1, scale_factor=2, mode="nearest")
        iconv0 = self.iconv0(self.upconv0(up0))
        disp1 = self.disp1(iconv0)  # full resolution (1x)

        return [disp1, disp2, disp3, disp4]


class ResNetDepthNet(nn.Module):
    def __init__(self, num_scales: int = 4, pretrained: bool = True, bias_init: float = -4.5):
        super().__init__()
        self.num_scales = num_scales
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder = DepthDecoder(num_scales=num_scales, bias_init=bias_init)

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        features = self.encoder(x)
        outputs = self.decoder(features)
        if self.training:
            return outputs[: self.num_scales]
        return outputs[0]
