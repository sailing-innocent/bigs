#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array



def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0 

    C2W = np.linalg.inv(Rt) 
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def simple_sphere(center: np.array, radius: float, split_theta = 10, split_phi = 10, color = None, split_r = 1, pos_colored=False) -> BasicPointCloud:
    theta = np.linspace(0, 2 * np.pi, split_theta)
    phi = np.linspace(0, np.pi, split_phi)
    theta, phi = np.meshgrid(theta, phi)
    N = split_theta * split_phi * split_r
    r_step = radius / split_r
    points = np.zeros((N, 3))
    colors = np.ones_like(points) if color is None else np.tile(color, (points.shape[0], 1))
    
    for ri in range(split_r):
        r = r_step * (ri + 1)
        rx = np.sin(phi) * np.cos(theta)
        ry = np.sin(phi) * np.sin(theta)
        rz = np.cos(phi)
        if pos_colored:
            colors[ri * split_theta * split_phi:(ri + 1) * split_theta * split_phi, :] = np.tile(np.array([rx.flatten(), ry.flatten(), rz.flatten()]).T, (1, 1))
        else:
            colors[ri * split_theta * split_phi:(ri + 1) * split_theta * split_phi, :] = np.tile(color, (split_theta * split_phi, 1))
        x = center[0] + r * rx
        y = center[1] + r * ry
        z = center[2] + r * rz
        points[ri * split_theta * split_phi:(ri + 1) * split_theta * split_phi, 0] = x.flatten()
        points[ri * split_theta * split_phi:(ri + 1) * split_theta * split_phi, 1] = y.flatten()
        points[ri * split_theta * split_phi:(ri + 1) * split_theta * split_phi, 2] = z.flatten()
    
    
    normals = points - center
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points, colors, normals)

def simple_point()-> BasicPointCloud:
    points = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    normals = np.array([[0, 0, 1]])
    return BasicPointCloud(points, colors, normals)
