# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coordinate transforms between the native WAM latent and LIBERO's 7D SDE."""

import math

import torch
import torch.nn.functional as F


def _rotation_matrix(columns: torch.Tensor) -> torch.Tensor:
    """Project two column vectors to the nearest right-handed rotation."""
    a, b = columns.split(3, dim=-1)
    eps = torch.finfo(columns.dtype).eps
    norm_a = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    norm_b = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    normal = torch.linalg.cross(a, b)
    norm_normal = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)

    # The polar factor is nonunique for collinear columns. Choose a stable
    # normal perpendicular to the longer column. All-zero input picks identity.
    primary = torch.where(norm_a >= norm_b, a, b)
    eye = torch.eye(3, dtype=columns.dtype, device=columns.device)
    primary = torch.where(torch.maximum(norm_a, norm_b) > eps, primary, eye[0])
    primary = F.normalize(primary, dim=-1, eps=eps)
    auxiliary = eye[primary.abs().argmin(dim=-1)]
    fallback = torch.linalg.cross(primary, auxiliary)
    normal = torch.where(
        norm_normal > eps * (norm_a * norm_b).clamp_min(1), normal, fallback
    )
    n = F.normalize(normal, dim=-1, eps=eps)

    # At full rank n is the oriented plane normal. Under e1 = n x e0,
    # maximizing e0·a + e1·b selects e0 parallel to a + b x n. This gives
    # the same rotation as the native polar projection of [a,b,a x b].
    first = a + torch.linalg.cross(b, n)
    first = torch.where(
        torch.linalg.vector_norm(first, dim=-1, keepdim=True) > eps, first, primary
    )
    e0 = F.normalize(first, dim=-1, eps=eps)
    e1 = torch.linalg.cross(n, e0)
    return torch.stack((e0, e1, n), dim=-1)


def _rotation_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Return the principal axis-angle vector through a well-conditioned quaternion."""
    row0, row1, row2 = matrix.unbind(-2)
    r00, r01, r02 = row0.unbind(-1)
    r10, r11, r12 = row1.unbind(-1)
    r20, r21, r22 = row2.unbind(-1)
    # Each row is 4*q_component*(w,x,y,z). The largest diagonal selects
    # a nonzero row even at identity or a half turn, without an SVD backward.
    rows = torch.stack(
        (
            torch.stack((1 + r00 + r11 + r22, r21 - r12, r02 - r20, r10 - r01), -1),
            torch.stack((r21 - r12, 1 + r00 - r11 - r22, r01 + r10, r02 + r20), -1),
            torch.stack((r02 - r20, r01 + r10, 1 - r00 + r11 - r22, r12 + r21), -1),
            torch.stack((r10 - r01, r02 + r20, r12 + r21, 1 - r00 - r11 + r22), -1),
        ),
        dim=-2,
    )
    index = rows.diagonal(dim1=-2, dim2=-1).argmax(-1)
    select = index[..., None, None].expand(*index.shape, 1, 4)
    quat = F.normalize(rows.gather(-2, select).squeeze(-2), dim=-1)
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    vector = quat[..., 1:]
    half_angle = torch.atan2(
        torch.linalg.vector_norm(vector, dim=-1, keepdim=True), quat[..., :1]
    )
    return 2 * vector / torch.sinc(half_angle / math.pi)


def _rotation_columns(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula for the first two columns, including zero rotation."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    # Use the outer product identity for K^2 to avoid reduced-precision GEMM.
    skew_sq = vector.unsqueeze(-1) * vector.unsqueeze(-2) - angle.square() * eye
    matrix = (
        eye
        + torch.sinc(angle / math.pi) * skew
        + 0.5 * torch.sinc(angle / (2 * math.pi)).square() * skew_sq
    )
    return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def to_sde_action(
    action, rotation_offset, rotation_scale, axisangle_offset, axisangle_scale
):
    """Map normalized WAM [xyz,rot6d,gripper,pad] to normalized [xyz,rotvec,gripper]."""
    if action.shape[-1] < 10:
        raise ValueError(
            "LIBERO WAM actions must contain at least ten native coordinates."
        )
    action = action if action.dtype == torch.float64 else action.float()
    raw = action[..., 3:9] * rotation_scale.to(action) + rotation_offset.to(action)
    vector = _rotation_vector(_rotation_matrix(raw))
    rotation = (vector - axisangle_offset.to(action)) / axisangle_scale.to(action)
    return torch.cat((action[..., :3], rotation, action[..., 9:10]), dim=-1)


def from_sde_action(
    sample, template, rotation_offset, rotation_scale, axisangle_offset, axisangle_scale
):
    """Lift a normalized 7D sample to the network latent, with zero padding."""
    if (
        sample.shape[-1] != 7
        or template.shape[-1] < 10
        or sample.shape[:-1] != template.shape[:-1]
    ):
        raise ValueError(
            "Expected a 7D LIBERO sample and a matching padded WAM action template."
        )
    sample = sample if sample.dtype == torch.float64 else sample.float()
    vector = sample[..., 3:6] * axisangle_scale.to(sample) + axisangle_offset.to(sample)
    columns = _rotation_columns(vector)
    rotation = (columns - rotation_offset.to(sample)) / rotation_scale.to(sample)
    padding = sample.new_zeros((*sample.shape[:-1], template.shape[-1] - 10))
    return torch.cat((sample[..., :3], rotation, sample[..., 6:7], padding), dim=-1)
