# -*- coding: utf-8 -*-
# @file cameras.py
# @brief Camera class for 3D rendering
# @author sailing-innocent
# @date 2025-04-04
# @version 1.0
# ---------------------------------
# First Edition by https://github.com/graphdeco-inria/gaussian-splatting
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from enum import Enum 
from PIL import Image
import os 

# The Camera Model
# -----------------------------------------------
# ------- Y DOWN | Z FRONT | X RIGHT ------------
# -----------------------------------------------
# view matrix = [R^T|T]
class Camera:
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, 
                 gt_alpha_mask = None,
                 image_name = None, 
                 uid = None,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, 
                 data_device = "cuda"):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = 800
        self.image_height = 800
        try:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        except:
            pass 

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        # World Coordinate to View Coordinate 

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # Projection 
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # Full Projection
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # Camera Center
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property 
    def debug_lines(self):
        tanfovx = np.tan(self.FoVx/2)
        tanfovy = np.tan(self.FoVy/2)
        znear = 1
        O = np.array([0, 0, 0])
        A = np.array([tanfovx/2, tanfovy/2, znear])
        B = np.array([-tanfovx/2, tanfovy/2, znear])
        C = np.array([-tanfovx/2, -tanfovy/2, znear])
        D = np.array([tanfovx/2, -tanfovy/2, znear])
        # -> 6, 3 stack 
        points = np.stack([O, A, B, C, D], axis=0)
        # transform to world space
        R = self.R.T
        T = - self.R @ self.T
        points = points @ R + T
        # points to lines OA, OB, OC, OD, AB, BC, CD, DA
        lines = np.array([
            [points[0], points[1]],
            [points[0], points[2]],
            [points[0], points[3]],
            [points[0], points[4]],
            [points[1], points[2]],
            [points[2], points[3]],
            [points[3], points[4]],
            [points[4], points[1]]
        ])
        return lines.flatten().tolist()

    def get_style_img(self, style_img_dir):
        style_img_name = os.path.join(style_img_dir, self.image_name + ".png")
        img = Image.open(style_img_name).convert("RGB")
        return img

    # from LangSplat
    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)

        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1) # 512, h, w
       
        return point_feature.cuda(), mask.cuda()

def get_lookat_cam(pos: np.array, target: np.array, world_name: str = "blender", v_width = 800, v_height = 800):
    # the world_name defines the assumption of world
    up = np.array([0, 0, 1])
    if (world_name == "blender"):
        # blender use a z up, y front, x right world coordinate 
        up = np.array([0, 0, 1])
    elif (world_name == "colmap"):
        # colmap use a y down, z front, x right world coordinate
        up = np.array([0, -1, 0])
    else:
        raise ValueError("Unknown world name")
    
    # our camera coord is same as colmap
    cz = target - pos 
    cz = cz / np.linalg.norm(cz)
    cx = np.cross(cz, up)
    cx = cx / np.linalg.norm(cx)
    cy = np.cross(cz, cx)
    # 计算了相机三个坐标轴的世界坐标系

    R = np.zeros((3, 3))
    R[:, 0] = cx
    R[:, 1] = cy
    R[:, 2] = cz 
    # GS python里面的相机模型非常抽象地多transpose一下，然后transpose回去，所以nothing to do with R 
    T = - R.T @ pos

    # FoVx = 60 / 180 * np.pi
    FoVy = 60 / 180 * np.pi 
    FoVx = np.arctan(np.tan(FoVy / 2) * v_width / v_height) * 2

    image = torch.zeros((3, v_height,v_width), dtype=torch.float)
    return Camera(-1, R, T, FoVx, FoVy, image, None, None, None)

def get_round_cam(rot_center_point, rot_axis, eye_fix_point, theta, scale=3.0, world_name: str = "blender", v_width = 800, v_height = 800):
    pos2d = [np.cos(theta), np.sin(theta)]
    pos3d = np.zeros(3)
    x = np.array([1, 0, 0])
    rx = np.cross(x, rot_axis)
    rx = rx / np.linalg.norm(rx)
    ry = np.cross(rot_axis, rx)
    ry = ry / np.linalg.norm(ry)
    pos3d = rot_center_point + scale * (pos2d[0] * rx + pos2d[1] * ry)
    return get_lookat_cam(pos3d, eye_fix_point, world_name=world_name, v_width=v_width, v_height=v_height), pos3d 