# -*- coding: utf-8 -*-
# @file render.py
# @brief MY Rendering script
# @author sailing-innocent
# @date 2025-01-13
# @version 1.0
# ---------------------------------

import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from v_arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from module.vanilla_3dgs_render import render
from gaussian_featmark import gs_mark_debug
from gaussian_featmark import gs_mark
from utils.image_utils import feat_to_color_img
from utils.gaussian_utils import feat_to_color_gs 
from cent.lib.sailtorch.point import point_vis
import matplotlib.pyplot as plt 

@torch.no_grad
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, impl):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    N = gaussians.get_xyz.shape[0] 
    points = gaussians.get_xyz
    debug_lines = []
    feat = torch.zeros((N, 3 + 1), dtype=torch.float32, device="cuda")
    for idx, view in enumerate(tqdm(views, desc="Building Feat Process")):
        gt = view.original_image[0:3, :, :]
        feat_img = gt
        gs_mark(gaussians, view, feat_img, feat)
        debug_lines += view.debug_lines
    feat = feat.detach()
    feat = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    # color = feat_to_color_gs(feat)
    # point_vis(points, color, debug_lines, 3.0, 1600, 900)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        feat_img, radii = gs_mark_debug(gaussians, view, feat)
        colored_img = feat_to_color_img(feat_img)
        # colored_img = colored_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0,1)
        torchvision.utils.save_image(colored_img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, impl: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, impl)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, impl)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--impl", type=str, default="vanilla")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.impl)