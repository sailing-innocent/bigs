# -*- coding: utf-8 -*-
# @file gsd_2_teaser.py
# @brief Make a teaser
# @author sailing-innocent
# @date 2025-03-02
# @version 1.0
# ---------------------------------

from scene.gaussian_model import GaussianModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from bigs import gs_mark

# from bigs import gs_mark_debug
# from bigs import gs_mark_var
from lib.vanilla_3dgs_render import render, render_with_edit

# from config import get_teaser_json
from config import get_config
import PIL
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# import tqdm
import time
# from utils.gaussian_utils import feat_to_color_gs
# from utils.image_utils import read_img
# from utils.gaussian_utils import get_from_gs
# from utils.gsd_utils import get_cam_list_round, get_cam_list_dataset

from scene.cameras import get_lookat_cam


class Params:
    stride = 2
    radius = 20
    crt = 0.3
    bb = 0.1
    with_var = False
    var_crt = 100.0


from utils.sh_utils import eval_sh


def sh2rgb_gs(pc, viewpoint_camera):
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
        pc.get_features.shape[0], 1
    )
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp


def preprocess_edit(edit_img):
    H, W, C = edit_img.shape  # H, W, 4
    assert C == 4
    edit_mask = edit_img[:, :, 3] > 0.1
    edit_rgb = edit_img[:, :, 0:3] / 255.0
    edit_rgb[~edit_mask] = 0.0
    return edit_rgb.transpose(2, 0, 1), edit_mask


@torch.no_grad()
def apply_edit(name: str, scene_name: str, prt: Params):
    # gs_scene = get_teaser_json()
    gs_scene = get_config(name)[scene_name]
    # scene_name = gs_scene["scene_name"]
    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"
    res_dir = os.path.join(out_dir, "res")
    os.makedirs(res_dir, exist_ok=True)
    edit_dir = os.path.join(out_dir, "edit")
    world_name = gs_scene["world_name"]
    world_name = gs_scene["world_name"]

    # bg_color = [1, 1, 1] if gs_scene["white_background"] else [0, 0, 0]
    bg_color = [0, 0, 0]
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    lines = []
    torch.cuda.reset_peak_memory_stats()  # reset to track the maximum CUDA Memory
    gs = GaussianModel(3)
    gs.load_ply(ply_path, False)
    N = gs.get_xyz.shape[0]
    logger.info("Gaussians Num: ", str(N))

    # feat = torch.zeros((N, 3 + 1), dtype=torch.float32).cuda() # additional edit feature

    edit_0_f = edit_dir + "/edit0.png"
    edit_1_f = edit_dir + "/edit1.png"
    # Assume Directory like
    # <scene_name>/edit/
    #  - edit0.png
    #  - edit1.png
    # Then output
    # <scene_name>/res/
    #  - <scene_name>_lookat.png
    #  - <scene_name>_another.png
    #  - ...

    edit_0 = np.array(PIL.Image.open(edit_0_f))
    edit_1 = np.array(PIL.Image.open(edit_1_f))
    # plt.imshow(edit_0)
    # plt.show()
    logger.info(edit_0.shape)  # H, W, 4
    logger.info(edit_0.max())  # 255
    edit_0, _ = preprocess_edit(edit_0)
    edit_1, _ = preprocess_edit(edit_1)

    # H, W, C -> C, H, W
    w = edit_0.shape[2]
    h = edit_0.shape[1]
    logger.info("Image Size: ", str(w), str(h))

    # test_frames = [0, 50]

    # use to fix the camera position and direction
    default_cam_pos = np.array([-0.92, -1.11, -5.08])
    default_cam_dir = np.array([-0.05, 0.3, 0.95])
    another_cam_pos = np.array([0.92, 1.11, -5.08])
    another_cam_dir = np.array([0.05, 0.3, 0.95])

    target = default_cam_pos + default_cam_dir

    lookat_cam = get_lookat_cam(default_cam_pos, target, world_name, w, h)
    another_cam = get_lookat_cam(another_cam_pos, another_cam_dir, world_name, w, h)
    # render the lookat result
    lookat_img = render(lookat_cam, gs, bg_color)["render"]
    lookat_img_save = lookat_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    plt.imsave(os.path.join(res_dir, f"{scene_name}_lookat.png"), lookat_img_save)
    another_img = render(another_cam, gs, bg_color)["render"]
    another_img_save = another_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    plt.imsave(os.path.join(res_dir, f"{scene_name}_another.png"), another_img_save)

    # Premark to get the basic feature and weight
    feat = torch.zeros(N, 4).float().cuda()
    feat_img = lookat_img
    logger.info(feat_img.shape)  # (3, H, W)
    gs_mark(gs, lookat_cam, feat_img, feat)
    feat_img = another_img
    gs_mark(gs, another_cam, feat_img, feat)

    logger.info(
        "Peak Memory Allocated Before Run: " + str(torch.cuda.max_memory_allocated())
    )
    t = time.time()  # Start Tracking Time

    edit_feat = torch.zeros(N, 4).float().cuda()
    edit_img = torch.from_numpy(edit_0).float().cuda()

    gs_mark(gs, lookat_cam, edit_img, edit_feat)
    edit_gs = edit_feat[:, 1:] / (edit_feat[:, 0:1] + 1e-6)
    # normalize the color
    edit_norm = edit_gs.norm(dim=1, keepdim=True)
    edit_mask = edit_norm > 1e-1
    edit_feat[:, 0:1][~edit_mask] = 0.0  # clear the weight
    # merge to feat
    feat = feat + edit_feat.clone()
    edit_feat = torch.zeros(N, 4).float().cuda()

    editted_gs = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    edited = render(lookat_cam, gs, bg_color, 1.0, editted_gs)["render"]
    edited_save = edited.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    another_img = render(another_cam, gs, bg_color, 1.0, editted_gs)["render"]
    another_img_save = another_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    cur_t = time.time()
    logger.info("Time Cost: " + str(cur_t - t))
    t = cur_t
    logger.info(
        "Peak Memory Allocated After Run: " + str(torch.cuda.max_memory_allocated())
    )

    plt.imsave(os.path.join(res_dir, f"{scene_name}_edited0.png"), edited_save)
    plt.imsave(os.path.join(res_dir, f"{scene_name}_another0.png"), another_img_save)

    edit_img = torch.from_numpy(edit_1).float().cuda()
    gs_mark(gs, lookat_cam, edit_img, edit_feat)
    edit_gs = edit_feat[:, 1:] / (edit_feat[:, 0:1] + 1e-6)
    # normalize the color
    edit_norm = edit_gs.norm(dim=1, keepdim=True)
    edit_mask = edit_norm > 1e-1
    edit_feat[:, 0:1][~edit_mask] = 0.0  # clear the weight
    # merge to feat
    feat = feat + edit_feat.clone()
    edit_feat = torch.zeros(N, 4).float().cuda()

    editted_gs = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    edited = render(lookat_cam, gs, bg_color, 1.0, editted_gs)["render"]
    edited_save = edited.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    another_img = render(another_cam, gs, bg_color, 1.0, editted_gs)["render"]
    another_img_save = another_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    cur_t = time.time()
    logger.info("Time Cost: " + str(cur_t - t))
    t = cur_t
    logger.info(
        "Peak Memory Allocated After Run: " + str(torch.cuda.max_memory_allocated())
    )

    plt.imsave(os.path.join(res_dir, f"{scene_name}_edited1.png"), edited_save)
    plt.imsave(os.path.join(res_dir, f"{scene_name}_another1.png"), another_img_save)

    feat = feat + edit_feat.clone()
    edit_feat = torch.zeros(N, 4).float().cuda()

    edit_img = torch.from_numpy(edit_1).float().cuda()
    gs_mark(gs, lookat_cam, edit_img, edit_feat)
    edit_gs = edit_feat[:, 1:] / (edit_feat[:, 0:1] + 1e-6)
    # normalize the color
    edit_norm = edit_gs.norm(dim=1, keepdim=True)
    edit_mask = edit_norm > 1e-1
    edit_feat[:, 0:1][~edit_mask] = 0.0  # clear the weight
    feat = feat - edit_feat.clone()
    edit_feat = torch.zeros(N, 4).float().cuda()
    orig_gs = feat[:, 1:] / (feat[:, 0:1] + 1e-6)

    edited = render(lookat_cam, gs, bg_color, 1.0, orig_gs)["render"]
    edited_save = edited.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    another_img = render(another_cam, gs, bg_color, 1.0, orig_gs)["render"]
    another_img_save = another_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    cur_t = time.time()
    logger.info("Time Cost: " + str(cur_t - t))
    t = cur_t
    logger.info(
        "Peak Memory Allocated After Run: " + str(torch.cuda.max_memory_allocated())
    )

    plt.imsave(os.path.join(res_dir, f"{scene_name}_edited2.png"), edited_save)
    plt.imsave(os.path.join(res_dir, f"{scene_name}_another2.png"), another_img_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="gsd"
    )  # task name, config/<task_name>.json
    parser.add_argument("--scene", type=str, default="in2n_bear")
    parser.add_argument("--with_var", action="store_true")
    parser.add_argument("--stride", type=int, default=10)
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

    apply_edit(args.name, args.scene, prt)
