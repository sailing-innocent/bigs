# -*- coding: utf-8 -*-
# @file gsd_2_obj_removal.py
# @brief The Core Object Removal Qualitative Experiment
# @author sailing-innocent
# @date 2025-02-27
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
from config import get_langsplat_json
import tqdm 
import time 
from utils.gsd_utils import get_cam_list_dataset
import PIL 
import matplotlib.pyplot as plt 
from utils.lerf_utils import feat_img_to_color_img

class Params:
    stride = 2
    with_var = True

@torch.no_grad()
def clip_mark(scene_name: str, out: bool, query, prt: Params = Params(),level:int=0, debug: bool = False):
    config_json = get_langsplat_json()
    if scene_name not in config_json:
        raise ValueError(f"Scene {scene_name} not found in gsd json")
    gs_scene = config_json[scene_name]
    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"
    dataset_path = gs_scene["dataset_path"]
    lan_feat_path = os.path.join(dataset_path, "language_features")
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
            img_clip_feat, mask = cam.get_language_feature(lan_feat_path, level)

            print(img_clip_feat.shape) # 512, h, w
            img_clip_feat_np = img_clip_feat.cpu().numpy().transpose(1, 2, 0) # h, w, 512
            print(img_clip_feat_np.shape) # h, w, 512
            color_img_clip = feat_img_to_color_img(img_clip_feat_np) # h, w, 3
            plt.imshow(color_img_clip)
            plt.axis('off')
            plt.show()
            src_img = cam.image.cpu().numpy().transpose(1, 2, 0)
            plt.imshow(src_img)
            plt.axis('off')
            plt.show()
            # break 
            feats.append(img_clip_feat.cpu().numpy()) # store for later use
    if debug:
        return

    torch.cuda.reset_peak_memory_stats() # reset to track the maximum CUDA Memory
    
    gs = GaussianModel(-1)
    gs.load_ply(ply_path, False)
    N = gs.get_xyz.shape[0]
    print("Gaussians Num: ", str(N))

    # CLIP feature DIM 512 + 1 for weight
    feat = torch.zeros((N, 512 + 1), dtype=torch.float32).cuda()
    # test_frames = [0, 50]
    print("Peak Memory Allocated Before Run (GB): " + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    t = time.time() # Start Tracking Time

    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        # feat_data = feats[idx]
        # feat_img = torch.from_numpy(feat_data).float().cuda() # load on device
        feat_img, _ = cam.get_language_feature(lan_feat_path, level)
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
        out_dir = f"{project_home}/{scene_name}_stride_{stride}"
        level_dir = os.path.join(out_dir, f"{level}")
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(level_dir, exist_ok=True)
        feat_f = f"{level_dir}/feat.pt"
        torch.save(feat, feat_f)

    out()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene", type=str, default="lerf_teatime")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with_var", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()
    rand_seed = 42
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    prt = Params()
    prt.stride = args.stride
    prt.with_var = args.with_var
    clip_mark(args.scene, args.out, args.query, prt, args.level, args.debug)
    