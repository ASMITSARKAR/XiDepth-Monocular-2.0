import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BackprojectDepth(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.register_buffer("id_coords", torch.from_numpy(id_coords))

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        batch_size = depth.shape[0]

        if inv_K.dim() == 2:
            inv_K = inv_K.unsqueeze(0).repeat(batch_size, 1, 1)
        elif inv_K.dim() == 3 and inv_K.shape[0] != batch_size:
            inv_K = inv_K.repeat(batch_size, 1, 1)

        if inv_K.shape[-1] == 4:
            inv_K = inv_K[:, :3, :3]

        pix_coords = torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], dim=0
        ).unsqueeze(0).repeat(batch_size, 1, 1).to(device=depth.device, dtype=depth.dtype)

        ones = torch.ones(
            batch_size, 1, self.height * self.width, device=depth.device, dtype=depth.dtype
        )
        pix_coords = torch.cat([pix_coords, ones], dim=1)

        cam_points = torch.matmul(inv_K, pix_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        return torch.cat([cam_points, ones], dim=1)


class Project3D(nn.Module):
    def __init__(self, height: int, width: int, eps: float = 1e-7):
        super().__init__()
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points: torch.Tensor, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]

        if K.dim() == 2:
            K = K.unsqueeze(0).repeat(batch_size, 1, 1)
        elif K.dim() == 3 and K.shape[0] != batch_size:
            K = K.repeat(batch_size, 1, 1)

        if T.dim() == 2:
            T = T.unsqueeze(0).repeat(batch_size, 1, 1)
        elif T.dim() == 3 and T.shape[0] != batch_size:
            T = T.repeat(batch_size, 1, 1)

        if K.shape[-1] == 3:
            K_4x4 = torch.eye(4, device=K.device, dtype=K.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            K_4x4[:, :3, :3] = K[:, :3, :3]
            K = K_4x4

        P = torch.matmul(K, T)
        cam_points = torch.matmul(P, points)

        cam_z = cam_points[:, 2:3, :].clamp(min=1e-3)
        pix_coords = cam_points[:, :2, :] / (cam_z + self.eps)
        pix_coords = torch.clamp(pix_coords, min=-2.0 * self.width, max=2.0 * self.width)

        pix_coords = pix_coords.view(batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        pix_x = (pix_coords[..., 0] / (self.width - 1) - 0.5) * 2.0
        pix_y = (pix_coords[..., 1] / (self.height - 1) - 0.5) * 2.0
        return torch.stack([pix_x, pix_y], dim=-1)


def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(vec, p=2, dim=2, keepdim=True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    c = 1.0 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs, ys, zs = x * sa, y * sa, z * sa
    xc, yc, zc = x * c, y * c, z * c
    xyc, yzc, zxc = x * yc, y * zc, z * xc

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device, dtype=vec.dtype)
    rot[:, 0, 0] = torch.squeeze(x * xc + ca)
    rot[:, 0, 1] = torch.squeeze(xyc - zs)
    rot[:, 0, 2] = torch.squeeze(zxc + ys)
    rot[:, 1, 0] = torch.squeeze(xyc + zs)
    rot[:, 1, 1] = torch.squeeze(y * yc + ca)
    rot[:, 1, 2] = torch.squeeze(yzc - xs)
    rot[:, 2, 0] = torch.squeeze(zxc - ys)
    rot[:, 2, 1] = torch.squeeze(yzc + xs)
    rot[:, 2, 2] = torch.squeeze(z * zc + ca)
    rot[:, 3, 3] = 1.0
    return rot


def get_translation_matrix(translation_vector: torch.Tensor) -> torch.Tensor:
    t = torch.zeros(
        translation_vector.shape[0], 4, 4,
        device=translation_vector.device, dtype=translation_vector.dtype
    )
    t[:, 0, 0] = 1.0
    t[:, 1, 1] = 1.0
    t[:, 2, 2] = 1.0
    t[:, 3, 3] = 1.0
    t[:, :3, 3] = translation_vector.contiguous().view(-1, 3)
    return t


def transformation_from_parameters(
    axisangle: torch.Tensor,
    translation: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    if axisangle.dim() == 4:
        axisangle = axisangle.squeeze(2)
    if translation.dim() == 4:
        translation = translation.squeeze(2)

    rot = rot_from_axisangle(axisangle)
    trans = translation.squeeze(1)

    if invert:
        rot = rot.transpose(1, 2)
        trans = trans * -1.0

    t_mat = get_translation_matrix(trans)
    return torch.matmul(rot, t_mat) if invert else torch.matmul(t_mat, rot)
