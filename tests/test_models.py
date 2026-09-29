import torch
import pytest

from models import XiDepthNet, ResNetDepthNet, PoseNet, disp_to_depth


def test_xidepth_shapes():
    model = XiDepthNet(num_scales=4)
    model.train()
    x = torch.randn(2, 3, 192, 640)
    outputs = model(x)

    assert len(outputs) == 4
    assert outputs[0].shape == (2, 1, 192, 640)
    assert outputs[1].shape == (2, 1, 96, 320)
    assert outputs[2].shape == (2, 1, 48, 160)
    assert outputs[3].shape == (2, 1, 24, 80)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (2, 1, 192, 640)


def test_resnet_depth_shapes():
    model = ResNetDepthNet(num_scales=4, pretrained=False)
    model.train()
    x = torch.randn(2, 3, 192, 640)
    outputs = model(x)

    assert len(outputs) == 4
    assert outputs[0].shape == (2, 1, 192, 640)
    assert outputs[1].shape == (2, 1, 96, 320)
    assert outputs[2].shape == (2, 1, 48, 160)
    assert outputs[3].shape == (2, 1, 24, 80)


def test_initial_depth_calibration():
    # Verifies that disparity bias init prevents the 1.3m near-plane collapse from v1
    x = torch.randn(2, 3, 192, 640)

    xi = XiDepthNet()
    _, depth_xi = disp_to_depth(xi(x)[0])
    mean_depth_xi = depth_xi.mean().item()
    assert 7.0 <= mean_depth_xi <= 11.0, f"XiDepth initial depth {mean_depth_xi}m out of bounds"

    resnet = ResNetDepthNet(pretrained=False)
    _, depth_res = disp_to_depth(resnet(x)[0])
    mean_depth_res = depth_res.mean().item()
    assert 7.0 <= mean_depth_res <= 11.0, f"ResNet initial depth {mean_depth_res}m out of bounds"


def test_posenet_shapes():
    model = PoseNet(num_input_images=2, pretrained=False)
    x = torch.randn(2, 6, 192, 640)
    axisangle, translation = model(x)

    assert axisangle.shape == (2, 1, 1, 3)
    assert translation.shape == (2, 1, 1, 3)
    assert torch.abs(translation).max() < 0.1


def test_official_monodepth2_shapes():
    from models import OfficialMonodepth2
    model = OfficialMonodepth2(num_scales=4, pretrained=False)
    model.train()
    x = torch.randn(2, 3, 192, 640)
    outputs = model(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert outputs[0].shape == (2, 1, 192, 640)
    assert outputs[1].shape == (2, 1, 96, 320)
    assert outputs[2].shape == (2, 1, 48, 160)
    assert outputs[3].shape == (2, 1, 24, 80)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (2, 1, 192, 640)

