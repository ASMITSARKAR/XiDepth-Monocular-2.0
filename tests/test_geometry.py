import torch
import pytest

from utils.geometry import (
    BackprojectDepth,
    Project3D,
    transformation_from_parameters,
    rot_from_axisangle,
)


def test_reprojection_identity():
    height, width = 192, 640
    batch_size = 2

    backproject = BackprojectDepth(height, width)
    project = Project3D(height, width)

    depth = torch.ones(batch_size, 1, height, width) * 10.0
    k = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    k[:, 0, 0] = 350.0
    k[:, 1, 1] = 350.0
    k[:, 0, 2] = width / 2.0
    k[:, 1, 2] = height / 2.0

    inv_k = torch.inverse(k)
    identity_t = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)

    points = backproject(depth, inv_k)
    pix_coords = project(points, k, identity_t)

    # Grid coords in [-1, 1]
    assert pix_coords.shape == (batch_size, height, width, 2)
    assert not torch.isnan(pix_coords).any()
    assert (pix_coords[..., 0] >= -1.01).all() and (pix_coords[..., 0] <= 1.01).all()
    assert (pix_coords[..., 1] >= -1.01).all() and (pix_coords[..., 1] <= 1.01).all()


def test_transformation_composition():
    batch_size = 2
    axisangle = torch.zeros(batch_size, 1, 1, 3)
    axisangle[:, :, :, 1] = 0.05  # slight yaw
    translation = torch.zeros(batch_size, 1, 1, 3)
    translation[:, :, :, 2] = 0.5  # forward translation

    t = transformation_from_parameters(axisangle, translation)
    assert t.shape == (batch_size, 4, 4)
    assert not torch.isnan(t).any()
    assert torch.allclose(t[:, 3, 3], torch.ones(batch_size))
