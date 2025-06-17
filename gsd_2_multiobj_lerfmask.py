# -*- coding: utf-8 -*-
# @file gsd_2_obj_removal.py
# @brief The Core Object Removal Qualitative Experiment
# @author sailing-innocent
# @date 2025-02-27
# @version 1.0
# ---------------------------------

from scene.gaussian_model import GaussianModel
import torch
from cent.utils.camera import Camera as SailCamera
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from gaussian_featmark import gs_mark
from gaussian_featmark import gs_mark_debug
from gaussian_featmark import gs_mark_var

from module.vanilla_3dgs_render import render_masked
from config import get_gsd_json

import tqdm
from cent.utils.video.av import write_mp4
import time
from cent.lib.sailtorch.gs import gs_vis

from utils.gaussian_utils import feat_to_color_gs
from utils.image_utils import read_img_with_blur

from utils.gsd_utils import get_cam_list_round, get_cam_list_dataset
import PIL
import matplotlib.pyplot as plt


class Params:
    stride = 2
    radius = 0
    crt = 0.5
    bb = 0
    with_var = True
    var_crt = 0.2


@torch.no_grad()
def multi_object_removal(
    scene_name: str, out: bool, selected_objs, prt: Params = Params()
):
    gsd_json = get_gsd_json()
    if scene_name not in gsd_json:
        raise ValueError(f"Scene {scene_name} not found in gsd json")
    gs_scene = gsd_json[scene_name]

    ply_path = gs_scene["ply_path"]
    project_home = gs_scene["project_home"]
    out_dir = f"{project_home}/{scene_name}"

    # mask_dir = os.path.join(out_dir, "frames/masks")
    dataset_path = gs_scene["dataset_path"]
    mask_dir = os.path.join(dataset_path, "object_mask")
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

    # selected_objs = gs_scene["selected_objs"]
    # N_classes = len(selected_objs)
    N_classes = 256

    test_frames = range(0, N_frames, stride)

    # load masks
    masks = []
    support_ids = {}
    selected_id = 157
    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        mask_f = os.path.join(mask_dir, cam.image_name + ".png")
        mask = PIL.Image.open(mask_f)
        # mask = np.array(mask)
        mask_data = np.array(mask)

        # BEGIN: mask_data to 0-255
        # -------------------------------------------
        fillter = mask_data == selected_id
        mask_data[fillter] = 1
        mask_data[~fillter] = 0
        plt.imshow(mask_data)
        plt.show()
        break
        # -------------------------------------------
        # END: fillter

        # BEGIN: add unique mask_data to support_ids
        # ----------------------------------------------
        # unique_ids = np.unique(mask_data)
        # for id in unique_ids:
        #     if id not in support_ids:
        #         support_ids[id] = 1
        #     else:
        #         support_ids[id] += 1
        # continue
        # ----------------------------------------------
        # END: add unique mask_data to support_ids

        # one-hot encoding i --> [0, 0, ... , 1(i'th place), ... 0]
        mask_data = np.eye(N_classes, dtype=np.uint8)[mask_data]  # H, W, N_classes
        # reshape to N_classes, H, W
        mask_data = mask_data.transpose(2, 0, 1)
        masks.append(mask_data)

    # print("support_ids: ", support_ids)
    # sorted_ids = sorted(support_ids.items(), key=lambda x: x[1], reverse=True)
    # top_ids = sorted_ids[:10]
    # print("Top 10 ids: ", top_ids)
    return

    lines = []
    torch.cuda.reset_peak_memory_stats()  # reset to track the maximum CUDA Memory
    gs = GaussianModel(3)
    gs.load_ply(ply_path, False)
    pos = gs.get_xyz
    scales = gs.get_scaling
    rotqs = gs.get_rotation
    N = gs.get_xyz.shape[0]
    print("Gaussians Num: ", str(N))
    feat = torch.zeros((N, N_classes + 1), dtype=torch.float32).cuda()
    # test_frames = [0, 50]
    print("Peak Memory Allocated Before Run: " + str(torch.cuda.max_memory_allocated()))
    t = time.time()  # Start Tracking Time

    for idx, i in enumerate(test_frames):
        cam = cam_list[i]
        mask_data = masks[idx]
        feat_img = torch.from_numpy(mask_data).float().cuda()
        # feat_img = feat_img.unsqueeze(0)
        gs_mark(gs, cam, feat_img, feat)
        lines += cam.debug_lines

    w_mask = feat[:, 0] < 1e-4

    feat = feat[:, 1:] / (feat[:, 0:1] + 1e-6)
    feat[w_mask] = -background_bias

    if prt.with_var:
        feat_var = torch.zeros((N, N_classes + 1), dtype=torch.float32).cuda()
        for idx, i in enumerate(test_frames):
            cam = cam_list[i]
            mask_data = masks[idx]
            feat_img = torch.from_numpy(mask_data).float().cuda()
            # feat_img = feat_img.unsqueeze(0)
            gs_mark_var(gs, cam, feat_img, feat, feat_var)

        feat_var = feat_var[:, 1:] / (feat_var[:, 0:1] + 1e-6)

    # sync torch
    torch.cuda.synchronize()
    t = time.time() - t
    print(f"Time: {t}")
    print(
        "Peak Memory Allocated All (GB): "
        + str(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
    )

    if not out:
        return

    # ------------------ Output ------------------
    # feat = feat.flatten() # flatten to (N, )
    # if prt.with_var:
    #     feat_var = feat_var.flatten()

    # selected_obj = 47
    def out(selected_obj):
        ofeat = feat[:, selected_obj]
        if prt.with_var:
            ofeat_var = feat_var[:, selected_obj]
        feat_mask = ofeat > crt
        if prt.with_var:
            feat_mask = feat_mask & (ofeat_var < prt.var_crt)
        inv_mask = ~feat_mask
        img_list = []
        inv_img_list = []

        out_dir = f"{project_home}/{scene_name}_crt_{crt}_stride_{stride}_radius_{radius}_bb_{background_bias}"
        out_dir = out_dir + f"_var_{prt.var_crt}" if prt.with_var else out_dir
        out_dir = out_dir + f"_id_{selected_obj}"
        print(out_dir)

        os.makedirs(out_dir, exist_ok=True)
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
        torch.save(ofeat, feat_f)
        if prt.with_var:
            feat_var_f = f"{out_dir}/feat_var.pt"
            print("Save feat_var ", ofeat_var.shape)
            torch.save(ofeat_var, feat_var_f)

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

            out_feat_img, _ = gs_mark_debug(gs, cam, ofeat.unsqueeze(1))
            out_feat_img = (
                out_feat_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            )
            out_feat_img = out_feat_img.repeat(3, axis=2)
            plt.imsave(f"{feat_img_dir}/{i}.jpg", out_feat_img)

            if prt.with_var:
                out_feat_var_img, _ = gs_mark_debug(gs, cam, ofeat_var.unsqueeze(1))
                out_feat_var_img = (
                    out_feat_var_img.detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .clip(0, 1)
                )
                out_feat_var_img = out_feat_var_img.repeat(3, axis=2)
                plt.imsave(f"{feat_var_img_dir}/{i}.jpg", out_feat_var_img)

    for selected_obj in selected_objs:
        out(selected_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="mip360_kitchen")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--out-video", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with_var", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--crt", type=float, default=0.5)
    parser.add_argument("--bb", type=float, default=0.0)
    parser.add_argument("--var_crt", type=float, default=0.2)
    parser.add_argument("--selected_objs", type=int, nargs="+", default=[47])

    args = parser.parse_args()

    prt = Params()
    prt.stride = args.stride
    prt.radius = args.radius
    prt.crt = args.crt
    prt.bb = args.bb
    prt.with_var = args.with_var
    prt.var_crt = args.var_crt

    multi_object_removal(args.scene, args.out, args.selected_objs, prt)
