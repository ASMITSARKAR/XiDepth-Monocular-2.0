import torch
import pytest

from utils.loss import SSIM, compute_reprojection_loss, compute_smoothness_loss


def test_ssim_identity():
    ssim = SSIM()
    img = torch.rand(2, 3, 192, 640)
    loss = ssim(img, img)
    # Identical images should have SSIM distance close to 0
    assert loss.mean().item() < 1e-4


def test_reprojection_loss():
    pred = torch.rand(2, 3, 192, 640, requires_grad=True)
    target = torch.rand(2, 3, 192, 640)

    loss = compute_reprojection_loss(pred, target)
    assert loss.shape == (2, 1, 192, 640)
    assert loss.mean() > 0

    loss.mean().backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_smoothness_loss_constant_disparity():
    # A completely flat disparity map has zero gradient, so smoothness loss should be 0
    disp = torch.ones(2, 1, 192, 640) * 0.1
    img = torch.rand(2, 3, 192, 640)

    loss = compute_smoothness_loss(disp, img)
    assert loss.item() < 1e-6
