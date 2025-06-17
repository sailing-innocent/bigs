# -*- coding: utf-8 -*-
# @file gsd_render_views_nvos.py
# @brief Render views for NVOS Dataset
# @author sailing-innocent
# @date 2025-01-23
# @version 1.0
# ---------------------------------

import os 
import argparse
import torch 
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import  cameraList_from_camInfos
from config import get_nvos_json 
from tqdm import tqdm
from module.vanilla_3dgs_render import render
import matplotlib.pyplot as plt

def render_views(scene: str, debug: bool):
    nvos_json = get_nvos_json()
    scene_config = nvos_json[scene]
    llff_dataset_root = nvos_json["llff_dataset_root"]
    ply_path = scene_config["ply_path"]
    # fetch scribble_view and ref_view
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    scene_name = scene_config["scene_name"]
    llff_dataset = os.path.join(llff_dataset_root, scene_name)
    
    out_dir = f"data/mid/gsd/{scene}"
    frame_dir = f"{out_dir}/frames" 
    os.makedirs(frame_dir, exist_ok=True)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene_info = readColmapSceneInfo(llff_dataset, None, False)
    cam_list = cameraList_from_camInfos(scene_info.train_cameras, 1, -1)

    for i, view in tqdm(enumerate(cam_list)):
        rendering = render(view, gaussians, background)["render"]
        rendering_np = rendering.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        plt.imsave(f"{frame_dir}/{i}.jpg", rendering_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="horns_center")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    render_views(args.scene, args.debug)