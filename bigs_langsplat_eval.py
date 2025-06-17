# -*- coding: utf-8 -*-
# @file bigs_langsplat_eval.py, modified from the LangSplat repo
# @brief Evaluate The Feature 
# @author sailing-innocent
# @date 2025-04-07
# @version 1.0
# ---------------------------------
from __future__ import annotations
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

from tqdm import tqdm 
import time 
from utils.gsd_utils import get_cam_list_dataset
import PIL 
import matplotlib.pyplot as plt 
from utils.lerf_utils import eval_gt_lerfdata
import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import lerf_eval.colormaps as colormaps 

from lerf_eval.openclip_encoder import OpenCLIPNetwork
from utils.lerf_utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

class Params:
    stride = 2
    crt = 0.5
    with_var = True
    var_crt = 0.2

import open_clip

def prob_img_to_color_img(_prob_img):
    # H, W, 1 -> H, W, 3
    # Normalize the probability image to [0, 1]
    _prob_img = _prob_img - _prob_img.min()
    if _prob_img.max() > 0:
        _prob_img = _prob_img / _prob_img.max()
    # repeat the probability image to 3 channels
    color_img = np.repeat(_prob_img, 3, axis=-1)
    return color_img

def feat_img_to_color_img(_feat_img):
    """
    Visualizes a feature image of shape (H, W, 512) using PCA to project to 3 channels.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    H, W, C = _feat_img.shape
    # Flatten the feature image to (H*W, C)
    feat = _feat_img.reshape(-1, C)
    
    # Apply PCA to reduce to 3 dimensions.
    pca = PCA(n_components=3, random_state=42)
    feat_pca = pca.fit_transform(feat)
    
    # Reshape back to (H, W, 3)
    feat_pca = feat_pca.reshape(H, W, 3)
    
    # Normalize the output to the range [0, 1] for visualization.
    feat_pca = feat_pca - feat_pca.min()
    if feat_pca.max() > 0:
        feat_pca = feat_pca / feat_pca.max()
        
    return feat_pca 


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape # n_levels, n_query, h, w

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        # foreach prompt
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list

def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)     
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num

@torch.no_grad()
def clip_mark(scene_name: str, out: bool, query, prt: Params = Params(), debug: bool = False, pre_render: bool = False):
    levels = [1,2,3]
    mask_thresh = 0.4
    # level = levels[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    config_json = get_langsplat_json()
    if scene_name not in config_json:
        raise ValueError(f"Scene {scene_name} not found in gsd json")
    gs_scene = config_json[scene_name]
    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"
    dataset_path = gs_scene["dataset_path"]
    lan_feat_path = os.path.join(dataset_path, "language_features")
    gt_path = gs_scene['gt_path']
    
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(gt_path, out_dir)

    print(image_paths)
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    stride = prt.stride
    crt = prt.crt
    cam_list = get_cam_list_dataset(gs_scene["dataset_path"])
    out_dir = f"{project_home}/{scene_name}_stride_{stride}"

    if pre_render:
        gs = GaussianModel(-1)
        gs.load_ply(ply_path, False)
        # Pre-render the Feat Image 
        for level in levels:
            level_dir = os.path.join(out_dir, f"{level}")
            feat_f = f"{level_dir}/feat.pt"
            feat = torch.load(feat_f)

            for id in tqdm(eval_index_list):
                cam = cam_list[id]
                # print(cam.image_name)
                out_feat_img, _ = gs_mark_debug(gs, cam, feat) 
                # print(out_feat_img.shape) # F, H, W
                out_feat_img = out_feat_img.detach().cpu().numpy().transpose(1,2,0) # H, W, F
                np.save(os.path.join(level_dir, f"{id}.npy"), out_feat_img)
                if out:
                    # output the PCA Visualization
                    gt, mask = cam.get_language_feature(lan_feat_path, level)
                    gt = gt.cpu().numpy().transpose(1,2,0) # H, W, 512
                    # concatenate the gt and out_feat_img vertically (2 * H, W, 512)
                    out_feat_img = np.concatenate((gt, out_feat_img), axis=0) # 2H, W, 512
                    out_pca = feat_img_to_color_img(out_feat_img)
                    # split into two images
                    out_pca_gt = out_pca[:image_shape[0], :, :]
                    out_pca_feat = out_pca[image_shape[0]:, :, :]
                    out_pca_gt_path = os.path.join(level_dir, f"{id}_gt.png")
                    out_pca_feat_path = os.path.join(level_dir, f"{id}_feat.png")
                    plt.imsave(out_pca_gt_path, out_pca_gt)
                    plt.imsave(out_pca_feat_path, out_pca_feat)
        return 

    compressed_sem_feats = np.zeros((len(levels), len(eval_index_list), *image_shape, 512), dtype=np.float32)
    for i in range(len(levels)):
        level_dir = os.path.join(out_dir, f"{levels[i]}")
        feat_paths_lvl = sorted(glob.glob(os.path.join(level_dir, '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        print(f"feat_paths_lvl: {feat_paths_lvl}")
        for j, idx in enumerate(eval_index_list):
            feat_img = np.load(feat_paths_lvl[j])
            compressed_sem_feats[i][j] = feat_img
            # compressed_sem_feats[i][j] = feat_img.transpose(1,2,0)

    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0

    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(out_dir) / 'res' / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...] # the stored low-dim feature
        restored_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        
        c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options)


        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

        acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
        acc_num += acc_num_img

    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    print(f'trunc thresh: {mask_thresh}')
    print(f"iou chosen: {mean_iou_chosen:.4f}")
    print(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    print("Localization accuracy: " + f'{acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="lerf_teatime")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pre_render", action="store_true")
    parser.add_argument("--with_var", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crt", type=float, default=0.5)
    parser.add_argument("--var_crt", type=float, default=.2)
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()
    
    rand_seed = 42
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    prt = Params()
    prt.stride = args.stride
    prt.crt = args.crt
    prt.with_var = args.with_var
    prt.var_crt = args.var_crt
    clip_mark(args.scene, args.out, args.query, prt, args.debug, args.pre_render)
    