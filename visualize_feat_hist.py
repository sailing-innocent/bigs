# -*- coding: utf-8 -*-
# @file visualize_feat.py
# @brief Visualize the Feature Data
# @author sailing-innocent
# @date 2025-03-24
# @version 1.0
# ---------------------------------

import argparse 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import torch 

def visualize_feat(scene: str, no_label: bool = False, fontsize: int = 16):
    """Visualize the Feature Data"""
    # base_dir = f"data/result/bigs/multi_obj_removal/multi_obj_removal"

    base_dir = f"data/result/bigs/obj_removal/obj_removal"
    render_img_dir = f"data/result/bigs/multi_obj_removal/multi_obj_removal"
    out_dir = "data/result/bigs/bigs_vis_feat/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    fig_prefix = "fig_bigs_vis"
    feat_prefix = "feat"
    feat_var_prefix = "feat_var"

    scenes = [
        "horns_center",
        "mip360_bonsai",
        "mip360_garden",
        "mip360_kitchen",
        "orchids",
        "in2n_bear"
    ]
    if scene != "all":
        scenes = [scene]

    postfix = "_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2"
    dir_list = [scene + postfix for scene in scenes]

    for i, d in enumerate(dir_list):
        s = scenes[i]
        res_dir = base_dir + "/" + d
        feat_img_dir = f"{res_dir}/feat_imgs/"
        feat_dir = f"{res_dir}/feat.pt"
        feat_var_dir = f"{res_dir}/feat_var.pt"
        feat = torch.load(feat_dir)
        feat_var = torch.load(feat_var_dir)
        # draw the histogram of the feature (0, 0.1, 1)        
        plt.figure(figsize=(12, 6))
        
        out_feat_img_path = out_dir + "/" + fig_prefix + "_" + feat_prefix + "_" + s + ".pdf"
        out_feat_var_img_path = out_dir + "/" + fig_prefix + "_" + feat_var_prefix + "_" + s + ".pdf"

        # save the feature hist
        plt.figure(figsize=(12, 6))
        plt.hist(feat.cpu().numpy().flatten(), bins=20, color='blue', alpha=0.7, label='Feature')
        if not no_label:
            plt.title(f"Feature Histogram for {s}", fontsize=fontsize+4)
            plt.xlabel("Feature Value", fontsize=fontsize)
            plt.ylabel("Frequency", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_offset_text().set_fontsize(fontsize-2)
        plt.savefig(out_feat_img_path, bbox_inches='tight')
        plt.close()

        print(f"Saved feature histogram for {s} to {out_feat_img_path}")

        # save the feature variance hist
        plt.figure(figsize=(12, 6))
        plt.hist(feat_var.cpu().numpy().flatten(), bins=20, color='red', alpha=0.7, label='Feature Variance')
        if not no_label:
            plt.title(f"Feature Variance Histogram for {s}", fontsize=fontsize+4)
            plt.xlabel("Feature Variance Value", fontsize=fontsize)
            plt.ylabel("Frequency", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_offset_text().set_fontsize(fontsize-2)
        plt.savefig(out_feat_var_img_path, bbox_inches='tight')
        plt.close()

        print(f"Saved feature variance histogram for {s} to {out_feat_var_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the Feature Data")
    parser.add_argument("--scene", type=str, default="all", help="The scene to visualize")
    parser.add_argument("--no_label", action="store_true", help="Do not show the label")
    parser.add_argument("--fontsize", type=int, default=16, help="Font size for labels and text")
    args = parser.parse_args()
    visualize_feat(args.scene, args.no_label, args.fontsize)