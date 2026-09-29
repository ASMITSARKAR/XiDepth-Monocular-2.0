from collections import OrderedDict
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Conv3x3(nn.Module):
    """Layer to pad and convolve input, exactly as in official Monodepth2."""
    def __init__(self, in_channels: int, out_channels: int, use_refl: bool = True):
        super().__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU, exactly as in official Monodepth2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.conv(x))


class OfficialResnetEncoder(nn.Module):
    """Official Monodepth2 ResNet-18 Encoder."""
    def __init__(self, num_layers: int = 18, pretrained: bool = False):
        super().__init__()
        assert num_layers == 18, "Only ResNet-18 supported for official weights"
        self.num_ch_enc = [64, 64, 128, 256, 512]

        resnet = models.resnet18(weights=None)
        self.encoder = nn.Module()
        self.encoder.conv1 = resnet.conv1
        self.encoder.bn1 = resnet.bn1
        self.encoder.relu = resnet.relu
        self.encoder.maxpool = resnet.maxpool
        self.encoder.layer1 = resnet.layer1
        self.encoder.layer2 = resnet.layer2
        self.encoder.layer3 = resnet.layer3
        self.encoder.layer4 = resnet.layer4

    def forward(self, input_image: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features


class OfficialDepthDecoder(nn.Module):
    """Official Monodepth2 Depth Decoder with skip connections."""
    def __init__(self, num_ch_enc: List[int] = [64, 64, 128, 256, 512], scales: range = range(4)):
        super().__init__()
        self.num_output_channels = 1
        self.use_skips = True
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: List[torch.Tensor]) -> dict:
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return outputs


class OfficialMonodepth2(nn.Module):
    """Complete official Monodepth2 model wrapper."""
    def __init__(self):
        super().__init__()
        self.encoder = OfficialResnetEncoder(18, pretrained=False)
        self.decoder = OfficialDepthDecoder(self.encoder.num_ch_enc)

    def load_pretrained(self, encoder_path: str, decoder_path: str):
        enc_dict = torch.load(encoder_path, map_location="cpu")
        dec_dict = torch.load(decoder_path, map_location="cpu")

        # Load encoder weights
        enc_model_dict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in enc_dict.items() if k in enc_model_dict})

        # Load decoder weights
        self.decoder.load_state_dict(dec_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[("disp", 0)]
