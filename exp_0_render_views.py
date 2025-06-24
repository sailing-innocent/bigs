# -*- coding: utf-8 -*-
# @file gsd_0_render_views_ply.py
# @brief Render Views According to Standalone PLY Files
# @author sailing-innocent
# @date 2025-02-27
# @version 1.0
# ---------------------------------

from scene.gaussian_model import GaussianModel
from lib.vanilla_3dgs_render import render
from config import get_config
import os
import tqdm
import torch
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBUG_TOOLS_READY = False
try:
    from lib.sail import point_vis
    from lib.sail import gs_vis

    DEBUG_TOOLS_READY = True
except ImportError:
    logger.error("Debug tools not available. Please install lib.sail to use point_vis.")

from utils.gaussian_utils import get_from_gs
from utils.gsd_utils import get_cam_list_round, get_cam_list_dataset


@torch.no_grad
def render_video(name: str, scene: str, debug: bool = False):
    gs = GaussianModel(3)
    gs_scene = get_config(name)[scene]
    ply_path = gs_scene["ply_path"]
    gs.load_ply(ply_path)
    N_frames = 90
    out_dir = f"data/mid/gsd/{scene}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    frame_dir = f"{out_dir}/frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    img_list = []
    world_name = gs_scene["world_name"]
    use_dataset = gs_scene["use_dataset"]

    bg_color = [1, 1, 1] if gs_scene["white_background"] else [0, 0, 0]
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    debug_lines = []

    if use_dataset:
        cam_list = get_cam_list_dataset(gs_scene["dataset_path"])
    else:
        cam_list = get_cam_list_round(gs_scene["cam_config"], world_name, N_frames)

    for i in tqdm.tqdm(range(N_frames)):
        cam = cam_list[i]
        if debug:
            debug_lines += cam.debug_lines
            continue
        rendering = render(cam, gs, bg_color)["render"]
        # CHW -> HWC
        rendering_np = rendering.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        plt.imsave(f"{frame_dir}/{i}.jpg", rendering_np)
        img_list.append(rendering_np)

    if debug:
        if not DEBUG_TOOLS_READY:
            logger.error("Debug tools are not ready. Cannot visualize points.")
            return
        points, colors, scales, rotqs = get_from_gs(gs)
        # point_vis(points, colors, debug_lines, 3.0, 1600, 900)
        gs_vis(points, colors, scales, rotqs)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="gsd"
    )  # task name, config/<task_name>.json
    parser.add_argument("--scene", type=str, default="mip360_kitchen")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    render_video(args.name, args.scene, args.debug)
