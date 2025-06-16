# -*- coding: utf-8 -*-
# @file image_utils.py
# @brief Image Utils based on https://team.inria.fr/graphdeco
# @author sailing-innocent
# @date 2025-02-28
# @version 1.0
# ---------------------------------

import torch
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os 

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def feat_to_color_img(feat_img: torch.Tensor):
    assert len(feat_img.shape) == 3 # F, H, W
    F = feat_img.shape[0]
    if (F == 1):
        # one channel
        colored_feat = torch.cat([feat_img, feat_img, feat_img], dim=0)
    elif (F == 3):
        # three channel
        colored_feat = feat_img
    return colored_feat.clone()

def read_img(image_path, gray=False):
    image = None
    # check if the is valid path
    if not os.path.exists(image_path):
        print(f"Invalid image path: {image_path}")
        return image
    try:
        image = Image.open(image_path)
        if gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    return image

def read_img_with_blur(image_path, radius=0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # blur the edge
    if radius > 0:
        kernel = np.ones((radius, radius), np.float32) / (radius * radius)
        image = cv2.filter2D(image, -1, kernel)
    return image

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
