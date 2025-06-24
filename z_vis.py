# -*- coding: utf-8 -*-
# @file vis.py
# @brief The Visualization Script
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

from utils.blender_utils import blender_executive, create_basic_camera
from scene.cameras import get_lookat_cam, get_round_cam
import numpy as np
import bpy
from mathutils import Matrix
import argparse
import os
import torch
from cent.lib.sailtorch.gs import gs_vis
from cent.lib.sailtorch.point import point_vis
from utils.gaussian_utils import get_from_gs_source, default_scene
from scene.dataset_readers import readNerfSyntheticInfo, readNerfStudioInfo
from utils.camera_utils import cameraList_from_camInfos


@blender_executive
def vis_gs_camera(root):
    pos = np.array([-1, -1, 2])
    target = np.array([0, 0, 0])
    cam = get_lookat_cam(pos, target, world_name="blender")
    bcam1 = create_basic_camera(name="Gaussian Camera")
    trans = cam.world_view_transform.cpu().numpy().transpose()
    trans = np.linalg.inv(trans)
    bcam1.matrix_world = Matrix(trans)
    # bcam1.location = pos.tolist()
    print(cam.camera_center.cpu().numpy())


@blender_executive
def vis_gs_round_cams(root):
    axis = np.array([0, 0, 1])
    rot_center = np.array([1, 1, 1])
    eye_fix_point = np.array([0, 0, 0])
    for i in range(0, 360, 20):
        theta = i / 180 * np.pi
        cam, pos = get_round_cam(
            rot_center, axis, eye_fix_point, theta, scale=3.0, world_name="colmap"
        )
        bcam = create_basic_camera(name=f"FlipZNoFlip_{i}")
        trans = cam.world_view_transform.cpu().numpy().transpose()
        trans = np.linalg.inv(trans)
        bcam.matrix_world = Matrix(trans)


@blender_executive
def vis_cam_nerf_synthetic(root, data_dir):
    print("data_dir: ", data_dir)
    scene_info = readNerfSyntheticInfo(data_dir, True, True)
    cam_infos = scene_info.train_cameras
    cam_list = cameraList_from_camInfos(cam_infos, 1, -1)
    for i, cam in enumerate(cam_list):
        bcam = create_basic_camera(name=f"train_{i}")
        trans = cam.world_view_transform.cpu().numpy().transpose()
        trans = np.linalg.inv(trans)
        bcam.matrix_world = Matrix(trans)


@blender_executive
def vis_cam_nerf_studio(root, data_dir):
    print("data_dir: ", data_dir)
    scene_info = readNerfStudioInfo(data_dir, True, True)
    cam_infos = scene_info.train_cameras
    cam_list = cameraList_from_camInfos(cam_infos, 1, -1)
    for i, cam in enumerate(cam_list):
        bcam = create_basic_camera(name=f"train_{i}")
        trans = cam.world_view_transform.cpu().numpy().transpose()
        trans = np.linalg.inv(trans)
        bcam.matrix_world = Matrix(trans)


def vis_gs_ply(source):
    if os.path.exists(source):
        points, colors, scales, rotqs = get_from_gs_source(source)
    else:
        points, colors, scales, rotqs = default_scene()
    point_vis(points, colors, [], 3.0, 1600, 900)


def vis_gs_proxymesh(source):
    if os.path.exists(source):
        points, colors, scales, rotqs = get_from_gs_source(source)
    else:
        points, colors, scales, rotqs = default_scene()
    gs_vis(points, colors, scales, rotqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", type=str, default="vis_gs_cam")
    parser.add_argument(
        "--source",
        type=str,
        default="data/pretrained/gaussian/mip360_kitchen_30000.ply",
    )
    args = parser.parse_args()
    source = args.source

    if args.usage == "vis_gs_cam":
        vis_gs_camera("vis_gs_cam")
    elif args.usage == "vis_gs_round_cams":
        vis_gs_round_cams("vis_gs_round_cams")
    elif args.usage == "vis_gs_ply":
        vis_gs_ply(source)
    elif args.usage == "vis_cam_nerf_synthetic":
        vis_cam_nerf_synthetic(filename="vis_cam_nerf_synthetic", data_dir=source)
    elif args.usage == "vis_cam_nerf_studio":
        vis_cam_nerf_studio(filename="vis_cam_nerf_studio", data_dir=source)
    elif args.usage == "vis_gs_proxymesh":
        vis_gs_proxymesh(source)
