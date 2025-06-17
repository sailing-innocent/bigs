# -*- coding: utf-8 -*-
# @file exp_1_vis_mask.py
# @brief Visualize the Predicted Masks on RGB Frames
# @author sailing-innocent
# @date 2025-06-17
# @version 1.0
# ---------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_mask(mask_path, radius=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # blur the edge
    if radius > 0:
        kernel = np.ones((radius, radius), np.float32) / (radius * radius)
        mask = cv2.filter2D(mask, -1, kernel)
    return mask


def read_rgb_img(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def mask_vis(name: str, scene_name: str, radius: int):
    out_dir = f"data/mid/{name}/{scene_name}"
    frame_dir = os.path.join(out_dir, "frames")
    mask_dir = os.path.join(out_dir, "frames/masks")
    frame_names = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
    N_frames = len(frame_names)
    vis_mask_dir = os.path.join(out_dir, "frames/vis_masks")
    if not os.path.exists(vis_mask_dir):
        os.makedirs(vis_mask_dir)
    with torch.no_grad():
        for i in tqdm(range(N_frames)):
            mask_f = f"{mask_dir}/mask_{i}_1.png"
            frame_f = f"{frame_dir}/{i}.jpg"
            mask = read_mask(mask_f, radius)
            frame = read_rgb_img(frame_f)
            plt.imshow(frame)
            plt.imshow(mask, alpha=0.5)
            plt.axis("off")
            frame = frame / 255.0
            mask = mask / 255.0
            blended = 0.5 * frame + 0.5 * mask[..., None]
            # save the figure
            # plt.savefig(f"{vis_mask_dir}/{i}.png")
            plt.imsave(f"{vis_mask_dir}/{i}.png", blended)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="gsd"
    )  # task name, config/<task_name>.json
    parser.add_argument("--scene", type=str, default="mip360_kitchen")
    parser.add_argument("--radius", type=int, default=0)
    args = parser.parse_args()
    mask_vis(args.name, args.scene, args.radius)
