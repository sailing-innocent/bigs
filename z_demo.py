# -*- coding: utf-8 -*-
# @file demo_render.py
# @brief Demo Render
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import argparse 
from scene.cameras import get_lookat_cam
import numpy as np 
import torch 
from config import get_reprod_config
from lib.vanilla_3dgs_render import render
from gaussian_featmark import gs_mark_debug, gs_mark
from utils.image_utils import feat_to_color_img
from utils.gaussian_utils import feat_to_color_gs 
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud, simple_sphere
import matplotlib.pyplot as plt

from lib.sail import point_vis

def demo_featmark(scene_name: str):
    config = get_reprod_config()
    scene_json = config[scene_name]
    demo_json = scene_json["demo"]
    pos = np.array(demo_json["pos"])
    target = np.array(demo_json["target"])
    cam = get_lookat_cam(pos, target, world_name="blender")

    ply_path = scene_json["ply_path"]
    gs = GaussianModel(3)
    gs.load_ply(ply_path)
    bg_color = [1, 1, 1] if scene_json["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # rendering = render(cam, gs, background)["render"]
    N = gs.get_xyz.shape[0]
    # feat = torch.ones((N, 3), dtype=torch.float32, device="cuda")
    # feat[:, 0] = 0.0
    H = 800
    W = 800
    feat_img = torch.ones((3, H, W), dtype=torch.float32, device="cuda")
    feat_img[0, :, :] = 0.0
    feat = torch.zeros((N, 3 + 1), dtype=torch.float32, device="cuda")
    gs_mark(gs, cam, feat_img, feat)
    points = gs.get_xyz 
    feat = feat.detach()
    feat = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    print(feat.shape)
    print(feat.min(), feat.max())
    color = feat_to_color_gs(feat)
    debug_lines = [] 
    debug_lines += cam.debug_lines
    point_vis(points, color, debug_lines, 3.0, 1600, 900)

def demo_featmark_debug(scene_name: str):
    config = get_reprod_config()
    scene_json = config[scene_name]
    demo_json = scene_json["demo"]
    pos = np.array(demo_json["pos"])
    target = np.array(demo_json["target"])
    cam = get_lookat_cam(pos, target, world_name="blender")

    ply_path = scene_json["ply_path"]
    gs = GaussianModel(3)
    gs.load_ply(ply_path)
    bg_color = [1, 1, 1] if scene_json["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # rendering = render(cam, gs, background)["render"]
    N = gs.get_xyz.shape[0]
    feat = torch.ones((N, 3), dtype=torch.float32, device="cuda")
    feat[:, 0] = 0.0
    feat_img, radii = gs_mark_debug(gs, cam, feat)
    print(feat_img.shape)
    colored_img = feat_to_color_img(feat_img)
    colored_img = colored_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0,1)
    plt.imshow(colored_img)
    plt.show()

def demo_render(scene_name: str):
    config = get_reprod_config()
    scene_json = config[scene_name]
    demo_json = scene_json["demo"]
    pos = np.array(demo_json["pos"])
    target = np.array(demo_json["target"])
    cam = get_lookat_cam(pos, target, world_name="blender")

    ply_path = scene_json["ply_path"]
    gs = GaussianModel(3)
    gs.load_ply(ply_path)

    bg_color = [1, 1, 1] if scene_json["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    rendering = render(cam, gs, background)["render"]
    rendering = rendering.detach().cpu().numpy().transpose(1, 2, 0).clip(0,1)
    plt.imshow(rendering)
    plt.show()

def save_debug_scene():
    gs = GaussianModel(3)
    center = np.array([0.0, 0.0, 0.0])
    sphere_pcd = simple_sphere(center, 1.0, pos_colored=True)
    gs.create_from_pcd(sphere_pcd, 0.1)
    gs.save_ply("data/assets/samples/gsplat_debug.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Render")
    parser.add_argument("--scene", type=str, default="nerf_blender_lego")
    parser.add_argument("--usage", type=str, default="render")
    args = parser.parse_args()
    scene_name = args.scene   
    if args.usage == "featmark_debug":
        demo_featmark_debug(scene_name)
    elif args.usage == "featmark":
        demo_featmark(scene_name)
    elif args.usage == "render":
        demo_render(scene_name)
    elif args.usage == "save_debug_scene":
        save_debug_scene()
    else:
        raise ValueError(f"Unknown usage {args.usage}")
