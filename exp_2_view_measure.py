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
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBUG_TOOLS_READY = False
try:
    from lib.sail import point_vis

    DEBUG_TOOLS_READY = True
except ImportError:
    logger.error("Debug tools not available. Please install lib.sail to use point_vis.")

from bigs import gs_mark

# from bigs import gs_mark_debug
# from bigs import gs_mark_var
from lib.vanilla_3dgs_render import render

from config import get_config

# import tqdm
# from cent.utils.video.av import write_mp4
import time

# from cent.lib.sailtorch.gs import gs_vis

from utils.gaussian_utils import feat_to_color_gs

# from utils.image_utils import read_img_with_blur
# from utils.gaussian_utils import get_from_gs
from utils.gsd_utils import get_cam_list_round, get_cam_list_dataset

from scene.cameras import get_lookat_cam


class Params:
    stride = 2
    radius = 20
    crt = 0.3
    bb = 0.1
    with_var = False
    var_crt = 100.0


@torch.no_grad()
def view_measure(name: str, scene_name: str, prt: Params):
    # gs_scene = get_teaser_json()
    gs_scene = get_config(name)[scene_name]

    # scene_name = gs_scene["scene_name"]
    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"

    mask_dir = os.path.join(out_dir, "frames/masks")
    stride = prt.stride
    world_name = gs_scene["world_name"]
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
    if not len(cam_list) > 0:
        logger.info("No Camera Found")
        return
    sample_cam = cam_list[0]
    w = sample_cam.image_width
    h = sample_cam.image_height

    for i in test_frames:
        mask_data = np.ones((h, w), dtype=np.float32)
        masks.append(mask_data)

    # test_frames = [0, 50]
    logger.info(
        "Peak Memory Allocated Before Run: " + str(torch.cuda.max_memory_allocated())
    )
    t = time.time()  # Start Tracking Time

    # use to fix the camera position and direction
    default_cam_pos = np.array([-0.92, -1.11, -5.08])
    default_cam_dir = np.array([-0.05, 0.3, 0.95])
    target = default_cam_pos + default_cam_dir

    lookat_cam = get_lookat_cam(default_cam_pos, target, world_name, w, h)
    # render the lookat result
    lookat_img = render(lookat_cam, gs, bg_color)["render"]
    lookat_img = lookat_img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    plt.imsave("lookat.png", lookat_img)

    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        mask_data = masks[idx]
        feat_img = torch.from_numpy(mask_data).cuda()
        feat_img = feat_img.unsqueeze(0)  # (H, W) -> (1, H, W)
        gs_mark(gs, cam, feat_img, feat)
        lines += cam.debug_lines

        temp_feat = feat.clone().detach()
        temp_feat = temp_feat[:, 1:] / (temp_feat[:, 0:1] + 1e-6)
        rate = temp_feat.sum() / N
        logger.info(f"Frame: {i}, Rate: {rate}")

        color = feat_to_color_gs(temp_feat)
        logger.info(f"Frame: {i}")

    temp_feat = feat.clone().detach()
    temp_feat = (temp_feat[:, 1:] - 1.0) / (temp_feat[:, 0:1] + 1e-6)
    rate = temp_feat.sum() / N
    logger.info(f"Frame: {i}, Rate: {rate}")

    color = feat_to_color_gs(temp_feat)
    logger.info(f"Frame: {i}")

    default_cam_pos = [-2.11, -9.66, -0.7]
    default_cam_dir = [0.24, 0.93, 0.28]
    view_width = 1024
    view_height = 768
    point_vis(pos, color, debug_lines=lines, point_size=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="gsd"
    )  # task name, config/<task_name>.json
    parser.add_argument("--scene", type=str, default="mip360_kitchen")
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

    view_measure(args.name, args.scene, prt)
