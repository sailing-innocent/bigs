# -*- coding: utf-8 -*-
# @file __init__.py
# @brief featmark, modified from gaussian cuda rasterizer @inria
# @author sailing-innocent
# @date 2025-02-18
# @version 1.0
# ---------------------------------

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
import torch.nn as nn
import torch
import numpy as np

# from torch.utils.cpp_extension import load
from . import _C


def gs_mark_debug(gs: GaussianModel, cam: Camera, feat: torch.Tensor):
    view_mat = cam.world_view_transform
    proj_mat = cam.full_proj_transform
    fovx = cam.FoVx
    fovy = cam.FoVy
    tanfovx = np.tan(0.5 * fovx)
    tanfovy = np.tan(0.5 * fovy)
    h = cam.image_height
    w = cam.image_width
    N = gs.get_xyz.shape[0]
    # output
    F = feat.shape[1]
    out_feat_img = torch.zeros([F, h, w], dtype=torch.float32).cuda()
    radii = torch.zeros([N], dtype=torch.int32).cuda()
    args = (
        gs.get_xyz,
        gs.get_opacity,
        gs.get_scaling,
        gs.get_rotation,
        feat,
        view_mat,
        proj_mat,
        float(tanfovx),
        float(tanfovy),
        out_feat_img,
        radii,
        True,
    )
    num_rendered = _C.gs_feat_mark_debug(*args)
    # print(num_rendered)
    return out_feat_img, radii


def gs_mark(gs: GaussianModel, cam: Camera, feat_img: torch.Tensor, feat: torch.Tensor):
    view_mat = cam.world_view_transform
    proj_mat = cam.full_proj_transform
    fovx = cam.FoVx
    fovy = cam.FoVy
    tanfovx = np.tan(0.5 * fovx)
    tanfovy = np.tan(0.5 * fovy)
    h = cam.image_height
    w = cam.image_width
    N = gs.get_xyz.shape[0]
    # output
    dbg_img = torch.zeros([3, h, w], dtype=torch.float32).cuda()
    radii = torch.zeros([N], dtype=torch.int32).cuda()
    F = feat_img.shape[0]
    args = (
        gs.get_xyz,
        gs.get_opacity,
        gs.get_scaling,
        gs.get_rotation,
        feat_img,
        view_mat,
        proj_mat,
        float(tanfovx),
        float(tanfovy),
        feat,
        dbg_img,
        radii,
        True,
    )
    num_rendered = _C.gs_feat_mark(*args)
    # print(num_rendered)
    return dbg_img, radii


def gs_mark_var(
    gs: GaussianModel,
    cam: Camera,
    feat_img: torch.Tensor,
    feat: torch.Tensor,
    feat_var: torch.Tensor,
):
    view_mat = cam.world_view_transform
    proj_mat = cam.full_proj_transform
    fovx = cam.FoVx
    fovy = cam.FoVy
    tanfovx = np.tan(0.5 * fovx)
    tanfovy = np.tan(0.5 * fovy)
    h = cam.image_height
    w = cam.image_width
    N = gs.get_xyz.shape[0]
    # output
    dbg_img = torch.zeros([3, h, w], dtype=torch.float32).cuda()
    radii = torch.zeros([N], dtype=torch.int32).cuda()
    F = feat_img.shape[0]
    args = (
        gs.get_xyz,
        gs.get_opacity,
        gs.get_scaling,
        gs.get_rotation,
        feat_img,
        view_mat,
        proj_mat,
        float(tanfovx),
        float(tanfovy),
        feat,
        feat_var,
        dbg_img,
        radii,
        True,
    )
    num_rendered = _C.gs_feat_mark_var(*args)
    # print(num_rendered)
    return dbg_img, radii
