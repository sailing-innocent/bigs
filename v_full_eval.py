# -*- coding: utf-8 -*-
# @file v_full_eval.py
# @brief The Full Evaluation Script
# @author sailing-innocent
# @date 2025-02-27
# @version 1.0
# ---------------------------------
import os
from argparse import ArgumentParser

# nerf_synthetic_scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
# nerf_synthetic_scenes = ["chair"]
nerf_synthetic_scenes = []
# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# mipnerf360_outdoor_scenes = ["bicycle", "garden", "stump"]
# mipnerf360_outdoor_scenes = ["bicycle"]
# mipnerf360_outdoor_scenes = ["flowers", "treehill"]
mipnerf360_outdoor_scenes = []
# mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# mipnerf360_indoor_scenes = ["kitchen"]
mipnerf360_indoor_scenes = []
# tanks_and_temples_scenes = ["truck", "train"]
# tanks_and_temples_scenes = ["truck"]
tanks_and_temples_scenes = []
# deep_blending_scenes = ["drjohnson", "playroom"]
deep_blending_scenes = []
# llff_scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "trex"]
# llff_scenes = ["horns"]
llff_scenes = []

lerf_mask_scenes = ["figurines", "ramen", "teatime"]
# lerf_scenes = []

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--mission_name", default="reprod_test")
parser.add_argument("--output_path", default="./data/mid/")

args, _ = parser.parse_known_args()
output_path = os.path.join(args.output_path, args.mission_name)
if not os.path.exists(output_path):
    os.makedirs(output_path)

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(llff_scenes)
all_scenes.extend(lerf_mask_scenes)
# all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", type=str, default="data/datasets/mip360")
    parser.add_argument('--nerf_synthetic', "-ns", type=str, default="data/datasets/nerf_synthetic")
    parser.add_argument("--lerf_mask_dataset", "-lfm", type=str, default="data/datasets/lerf_mask")
    parser.add_argument("--tanksandtemples", "-tat", type=str, default="data/datasets/tanksandtemples")
    parser.add_argument("--llff", "-llff", type=str, default="data/datasets/llff")
    # parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        exec_args = "python v_train.py -s " + source + " -i images_4 -m " + output_path + "/" + scene + common_args
        print(exec_args)
        os.system(exec_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python v_train.py -s " + source + " -i images_2 -m " + output_path + "/" + scene + common_args)
    for scene in nerf_synthetic_scenes:
        source = args.nerf_synthetic + "/" + scene
        exec_args = "python v_train.py -s " + source + " -m " + output_path + "/" + scene + common_args
        print(exec_args)
        os.system(exec_args)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        exec_args = "python v_train.py -s " + source + " -m " + output_path + "/" + scene + common_args
        print(exec_args)
        os.system(exec_args)
    for scene in llff_scenes:
        source = args.llff + "/" + scene
        exec_args = "python v_train.py -s " + source + " -i images_2 -m " + output_path + "/" + scene + common_args
        print(exec_args)
        os.system(exec_args)
    for scene in lerf_mask_scenes:
        source = args.lerf_mask_dataset + "/" + scene
        exec_args = "python v_train.py -s " + source + " -m " + output_path + "/" + scene + common_args
        print(exec_args)
        os.system(exec_args)
    
    # for scene in deep_blending_scenes:
    #     source = args.deepblending + "/" + scene
    #     os.system("python v_train.py -s " + source + " -m " + output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerf_synthetic + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in llff_scenes:
        all_sources.append(args.llff + "/" + scene)
    for scene in lerf_mask_scenes:
        all_sources.append(args.lerf_mask_dataset + "/" + scene)

    # for scene in deep_blending_scenes:
    #     all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        cmd = "python v_render.py --iteration 30000 -s " + source + " -m " + output_path + "/" + scene + common_args
        print(cmd)
        os.system(cmd)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + output_path + "/" + scene + "\" "

    os.system("python v_metrics.py -m " + scenes_string)
