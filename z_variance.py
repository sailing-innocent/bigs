import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from v_arguments import ModelParams, PipelineParams, OptimizationParams

from module.vanilla_3dgs_render import render 
from gaussian_featmark import gs_mark_debug
from gaussian_featmark import gs_mark
from gaussian_featmark import gs_mark_var
# from lib.featmark.jit import gs_mark
# from lib.featmark.jit import gs_mark_var
from utils.image_utils import feat_to_color_img
from utils.gaussian_utils import feat_to_color_gs 
from cent.lib.sailtorch.point import point_vis

import logging 
logging.getLogger().setLevel(logging.ERROR)

def training(opt_m, opt_o, opt_p):
    first_iter = 0
    torch.cuda.reset_peak_memory_stats()
    gaussians = GaussianModel(opt_m.sh_degree)
    scene = Scene(opt_m, gaussians)
    bg_color = [1, 1, 1] if opt_m.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_o.iterations), desc="Training progress")
    first_iter += 1
    N = gaussians.get_xyz.shape[0]
    debug_lines = []
    mask = torch.ones(N, dtype=torch.bool, device="cuda")
    batch_size = 20
    feat = torch.zeros((N, 3 + 1), dtype=torch.float32, device="cuda")
    feat_var = torch.zeros((N, 3 + 1), dtype=torch.float32, device="cuda")
    for iteration in range(first_iter, opt_o.iterations + 1):        
        iter_start.record()
        batch_stack = []
        feat_var.zero_() # reset the variance
        # gaussians.get_opacity[mask] = 0.0 # masked the iterations
        gaussians._opacity = gaussians.inverse_opacity_activation(gaussians.get_opacity * mask.unsqueeze(1).clone().to(torch.float32))
        for b in range(batch_size):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            view = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            gt_image = view.original_image.cuda()
            gs_mark(gaussians, view, gt_image, feat)
            batch_stack.append(view)
            # debug_lines += view.debug_lines

        out_feat = feat.detach().clone()
        out_feat = out_feat[:, 1:] / (out_feat[:, 0:1] + 1e-6)
        for b in range(batch_size):
            view = batch_stack[b]
            gt_image = view.original_image.cuda()
            gs_mark_var(gaussians, view, gt_image, out_feat, feat_var)
            debug_lines += view.debug_lines

        out_feat_var = feat_var.detach().clone()
        out_feat_var = out_feat_var[:, 1:] / (out_feat_var[:, 0:1] + 1e-6)
        # print(feat_var.max(), feat_var.min(), feat_var.mean())
        dist_var = out_feat_var.norm(dim=1)
        print(dist_var.max(), dist_var.min(), dist_var.mean())
        mask &= (dist_var < 0.2)
        # update progress bar
        print(mask.sum())
        progress_bar.update(1)

    # color = feat_to_color_gs(dist_var.unsqueeze(1)        
    out_feat = feat.detach().clone()
    weight = out_feat[:, 0]
    mask &= (weight > 0.1)
    out_feat = out_feat[:, 1:] / (out_feat[:, 0:1] + 1e-6)
    of_norm = out_feat.norm(dim=1)
    mask &= (of_norm > 0.2)
    color = feat_to_color_gs(out_feat)
    points = gaussians.get_xyz[mask]
    color = color[mask]
    point_vis(points, color, debug_lines, 3.0, 1600, 900)

    print("Peak Memory Allocated All (GB): " + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    with torch.no_grad():
        training(lp.extract(args), op.extract(args), pp.extract(args))
    # All done
    print("\nTraining complete.")
