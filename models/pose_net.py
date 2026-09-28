from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class PoseNet(nn.Module):
    def __init__(
        self,
        num_input_images: int = 2,
        num_frames_to_predict_for: Optional[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_input_images = num_input_images
        self.num_frames_to_predict_for = num_frames_to_predict_for or (num_input_images - 1)

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(weights=None)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = resnet.conv1.weight / 2.0
            self.conv1.weight[:, 3:] = resnet.conv1.weight / 2.0

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pose_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.pose_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pose_conv3 = nn.Conv2d(256, 6 * self.num_frames_to_predict_for, kernel_size=1)
        self.pose_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pose_relu(self.pose_conv1(x))
        x = self.pose_relu(self.pose_conv2(x))
        x = self.pose_conv3(x)

        out = x.mean(dim=(2, 3))
        # Small scale initialization prevents abrupt warp displacement early in training
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
