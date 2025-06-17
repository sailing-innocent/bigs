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

from bigs import gs_mark
from bigs import gs_mark_debug
from bigs import gs_mark_var

from lib.vanilla_3dgs_render import render_masked
from config import get_config

import tqdm
import time

from utils.gaussian_utils import feat_to_color_gs
from utils.image_utils import read_img_with_blur
from utils.gsd_utils import get_cam_list_round, get_cam_list_dataset
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBUG_TOOLS_READY = False
try:
    from lib.sail import gs_vis

    DEBUG_TOOLS_READY = True
except ImportError:
    logger.error("Debug tools not available. Please install lib.sail to use point_vis.")


class Params:
    stride = 2
    radius = 20
    crt = 0.3
    bb = 0.1
    with_var = False
    var_crt = 100.0


@torch.no_grad()
def object_removal(
    name: str,
    scene_name: str,
    out: bool,
    debug: bool,
    out_video: bool,
    prt: Params = Params(),
):
    gs_scene = get_config(name)[scene_name]

    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"

    mask_dir = os.path.join(out_dir, "frames/masks")
    stride = prt.stride
    radius = prt.radius
    crt = prt.crt
    background_bias = prt.bb

    world_name = gs_scene["world_name"]
    use_dataset = gs_scene["use_dataset"]
    N_frames = 90
    if use_dataset:
        cam_list = get_cam_list_dataset(gs_scene["dataset_path"])
    else:
        cam_list = get_cam_list_round(gs_scene["cam_config"], world_name, N_frames)

    bg_color = [1, 1, 1] if gs_scene["white_background"] else [0, 0, 0]
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    lines = []
    torch.cuda.reset_peak_memory_stats()  # reset to track the maximum CUDA Memory
    gs = GaussianModel(3)
    gs.load_ply(ply_path, False)
    pos = gs.get_xyz
    scales = gs.get_scaling
    rotqs = gs.get_rotation
    N = gs.get_xyz.shape[0]
    logger.info("Gaussians Num: ", str(N))

    feat = torch.zeros((N, 2), dtype=torch.float32).cuda()

    test_frames = range(0, N_frames, stride)
    # load masks
    masks = []

    for i in test_frames:
        mask_f = f"{mask_dir}/mask_{i}_1.png"
        mask = read_img_with_blur(mask_f, radius)
        mask_data = np.array(mask, dtype=np.float32) / 255.0
        mask_data = mask_data - background_bias
        masks.append(mask_data)

    # test_frames = [0, 50]
    logger.info(
        "Peak Memory Allocated Before Run: " + str(torch.cuda.max_memory_allocated())
    )
    t = time.time()  # Start Tracking Time

    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        mask_data = masks[idx]
        feat_img = torch.from_numpy(mask_data).cuda()
        feat_img = feat_img.unsqueeze(0)
        # logger.info(feat_img.shape)
        gs_mark(gs, cam, feat_img, feat)
        lines += cam.debug_lines

    w_mask = feat[:, 0] < 1e-4

    if debug:
        weight = feat[:, 0].clone().detach().unsqueeze(1)
        # gs_vis(pos, color, scales, rotqs, lines)
        weight = weight[~w_mask]
        logger.info(len(weight))
        logger.info(torch.max(weight))
        logger.info(torch.min(weight))
        logger.info(torch.median(weight))
        logger.info(torch.mean(weight))

    feat = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    feat[w_mask] = -background_bias

    t = time.time() - t
    logger.info(f"Time: {t}")
    logger.info(
        "Peak Memory Allocated All (GB): "
        + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
    )

    if debug:
        logger.info(torch.max(feat))
        logger.info(torch.min(feat))
        logger.info(torch.median(feat))
        logger.info(torch.mean(feat))

    if prt.with_var:
        feat_var = torch.zeros((N, 1 + 1), dtype=torch.float32).cuda()
        for idx, i in enumerate(test_frames):
            cam = cam_list[i]
            mask_data = masks[idx]
            feat_img = torch.from_numpy(mask_data).cuda()
            feat_img = feat_img.unsqueeze(0)
            gs_mark_var(gs, cam, feat_img, feat, feat_var)

        feat_var = feat_var[:, 1:] / (feat_var[:, 0:1] + 1e-6)

    if debug:
        if prt.with_var:
            dist_var = feat_var[:, 1:].norm(dim=1)
            logger.info(dist_var.max(), dist_var.min(), dist_var.mean())
            dist_var[dist_var < prt.var_crt] = 0.0
            color = feat_to_color_gs(dist_var.unsqueeze(1))
        else:
            color = feat_to_color_gs(feat)
        default_cam_pos = [-2.11, -9.66, -0.7]
        default_cam_dir = [0.2, 0.97, 0.11]
        view_width = 1024
        view_height = 1024
        gs_vis(
            pos,
            color,
            scales,
            rotqs,
            lines,
            view_width,
            view_height,
            default_cam_pos,
            default_cam_dir,
        )

    if not out or debug:
        return

    feat = feat.flatten()  # flatten to (N, )
    if prt.with_var:
        feat_var = feat_var.flatten()

    feat_mask = feat > crt
    if prt.with_var:
        feat_mask = feat_mask & (feat_var < prt.var_crt)
    inv_mask = ~feat_mask
    img_list = []
    inv_img_list = []

    out_dir = f"{project_home}/{scene_name}_crt_{crt}_stride_{stride}_radius_{radius}_bb_{background_bias}"
    out_dir = out_dir + f"_var_{prt.var_crt}" if prt.with_var else out_dir

    logger.info(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_dir = f"{out_dir}/imgs"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    inv_img_dir = f"{out_dir}/inv_imgs"
    if not os.path.exists(inv_img_dir):
        os.makedirs(inv_img_dir)
    feat_img_dir = f"{out_dir}/feat_imgs"
    os.makedirs(feat_img_dir, exist_ok=True)
    feat_var_img_dir = f"{out_dir}/feat_var_imgs"
    os.makedirs(feat_var_img_dir, exist_ok=True)

    feat_f = f"{out_dir}/feat.pt"
    torch.save(feat, feat_f)
    if prt.with_var:
        feat_var_f = f"{out_dir}/feat_var.pt"
        logger.info("Save feat_var ", feat_var.shape)
        torch.save(feat_var, feat_var_f)

    for i in tqdm.tqdm(range(N_frames)):
        cam = cam_list[i]

        img = render_masked(cam, gs, bg_color, feat_mask)["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        # rendering_np = rendering_np[::-1, :, :]
        plt.imsave(f"{img_dir}/{i}.jpg", img_np)
        img_list.append(img_np)

        inv_img = render_masked(cam, gs, bg_color, inv_mask)["render"]
        inv_img_np = inv_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        plt.imsave(f"{inv_img_dir}/{i}.jpg", inv_img_np)
        inv_img_list.append(inv_img_np)

        out_feat_img, _ = gs_mark_debug(gs, cam, feat.unsqueeze(1))
        out_feat_img = out_feat_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        out_feat_img = out_feat_img.repeat(3, axis=2)
        plt.imsave(f"{feat_img_dir}/{i}.jpg", out_feat_img)

        if prt.with_var:
            out_feat_var_img, _ = gs_mark_debug(gs, cam, feat_var.unsqueeze(1))
            out_feat_var_img = (
                out_feat_var_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            )
            out_feat_var_img = out_feat_var_img.repeat(3, axis=2)
            plt.imsave(f"{feat_var_img_dir}/{i}.jpg", out_feat_var_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="gsd")
    parser.add_argument("--scene", type=str, default="mip360_kitchen")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with_var", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--crt", type=float, default=0.5)
    parser.add_argument("--bb", type=float, default=0.0)
    parser.add_argument("--var_crt", type=float, default=0.2)

    args = parser.parse_args()

    prt = Params()
    prt.stride = args.stride
    prt.radius = args.radius
    prt.crt = args.crt
    prt.bb = args.bb
    prt.with_var = args.with_var
    prt.var_crt = args.var_crt

    object_removal(args.name, args.scene, args.out, args.debug, prt)
