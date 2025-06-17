# -*- coding: utf-8 -*-
# @file gaussian_utils.py
# @brief Gaussian Utils
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import torch
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def feat_to_color_gs(feat: torch.Tensor):
    assert len(feat.shape) == 2  # P, F
    F = feat.shape[1]
    if F == 1:
        # one channel
        colored_feat = torch.cat([feat, feat, feat], dim=1)
    elif F == 3:
        # three channel
        colored_feat = feat
    else:
        raise ValueError("feat_to_color: feat shape[1] must be 1 or 3")
    return colored_feat.clone()


def get_from_gs(gs: GaussianModel):
    # calculate SH in python
    camera_center = torch.tensor([0.0, 0.0, 0.0]).float().cuda()
    shs_view = gs.get_features.transpose(1, 2).view(-1, 3, (gs.max_sh_degree + 1) ** 2)
    dir_pp = gs.get_xyz - camera_center.repeat(gs.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gs.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    points = gs.get_xyz
    colors = colors_precomp
    scales = gs.get_scaling
    rotqs = gs.get_rotation
    return points, colors, scales, rotqs


def get_from_gs_source(source: str):
    gs = GaussianModel(3)
    gs.load_ply(source)
    return get_from_gs(gs)


def default_scene(N: int = 6):
    points = torch.randn(N, 3).cuda()
    color = torch.rand(N, 3).cuda()
    scale = torch.ones(3) * 0.5
    scales = torch.stack([scale for _ in range(N)], dim=0).cuda()
    theta = torch.tensor(45.0 * (3.141592653589793 / 180.0))  # 45 degrees in radians
    rotq = torch.tensor([1.0, 0.0, 0.0, 0.0])
    special_rotq = torch.tensor([torch.cos(theta / 2), 0.0, 0.0, torch.sin(theta / 2)])
    rotqs = torch.stack([rotq for _ in range(N - 1)], dim=0)
    rotqs = torch.cat([rotqs, special_rotq.unsqueeze(0)], dim=0)
    rotqs = rotqs.cuda()

    return points, color, scales, rotqs
