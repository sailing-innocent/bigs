# -*- coding: utf-8 -*-
# @file bigs_style.py
# @brief BIGS Style Transfer
# @author sailing-innocent
# @date 2025-04-08
# @version 1.0
# ---------------------------------

from scene.gaussian_model import GaussianModel
import torch
import numpy as np 
import matplotlib.pyplot as plt
import os 
import argparse 

from gaussian_featmark import gs_mark
from gaussian_featmark import gs_mark_debug
from gaussian_featmark import gs_mark_var
from config import get_style_json

import tqdm 
import time 
from utils.gsd_utils import get_cam_list_dataset
import PIL 
import matplotlib.pyplot as plt 

from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=42)

class Params:
    stride = 2

@torch.no_grad()
def style_mark(scene_name: str, out: bool, style:str, prt: Params = Params(),debug: bool = False):
    config_json = get_style_json()
    if scene_name not in config_json:
        raise ValueError(f"Scene {scene_name} not found in gsd json")
    gs_scene = config_json[scene_name]
    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"
    dataset_path = gs_scene["dataset_path"]
    style_feat_path = os.path.join(dataset_path, "styled", style)
    stride = prt.stride
    cam_list = get_cam_list_dataset(gs_scene["dataset_path"])
    N_frames = len(cam_list)
    bg_color = [1, 1, 1] if gs_scene["white_background"] else [0, 0, 0]
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    test_frames = range(0, N_frames, stride)
    print(f"Using {len(test_frames)}/{N_frames} frames")


    if (debug):
        feats = []
        for idx, i in enumerate(test_frames):
            cam = cam_list[i]
            img = cam.get_style_img(style_feat_path)
            img = np.array(img).astype(np.float32) / 255.0
            print("Image Shape: ", img.shape)
            print("Image max: ", img.max())
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            break 
    if debug:
        return

    torch.cuda.reset_peak_memory_stats() # reset to track the maximum CUDA Memory
    
    gs = GaussianModel(-1)
    gs.load_ply(ply_path, False)
    N = gs.get_xyz.shape[0]
    print("Gaussians Num: ", str(N))
    feat = torch.zeros((N, 3 + 1), dtype=torch.float32).cuda()
    print("Peak Memory Allocated Before Run (GB): " + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    t = time.time() # Start Tracking Time

    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        img = cam.get_style_img(style_feat_path)
        img = np.array(img).astype(np.float32) / 255.0 # H, W, 3
        feat_img = torch.from_numpy(img).float().cuda() # H, W, 3
        feat_img = feat_img.permute(2, 0, 1) # C, H, W
        gs_mark(gs, cam, feat_img, feat)
    w_mask = feat[:, 0] < 1e-4
    feat = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    feat[w_mask] = 0.0
    # sync torch
    torch.cuda.synchronize()
    t = time.time() - t
    print(f"Time: {t}")
    print("Peak Memory Allocated All (GB): " + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))        
    
    if (not out):
        return  
    
    def out():
        out_dir = f"{project_home}/style_{scene_name}_stride_{stride}"
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        style_dir = os.path.join(out_dir, style)
        os.makedirs(style_dir, exist_ok=True)
        feat_f = f"{style_dir}/feat.pt"
        torch.save(feat, feat_f)

        for cam in cam_list:
            out_feat_img,_ = gs_mark_debug(gs, cam, feat)
            out_feat_img = out_feat_img.permute(1, 2, 0).cpu().numpy() # H, W, C
            out_feat_img = (out_feat_img * 255).astype(np.uint8) # H, W, C
            out_feat_img = PIL.Image.fromarray(out_feat_img)
            # save to style_dir
            out_feat_img_name = os.path.join(style_dir, cam.image_name + ".png")
            out_feat_img.save(out_feat_img_name)
            print(f"Saved {out_feat_img_name}")

    out()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene", type=str, default="tt_truck")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--style", type=str, default="The_Kiss")
    args = parser.parse_args()

    prt = Params()
    prt.stride = args.stride
    style_mark(args.scene, args.out, args.style, prt, args.debug)
    