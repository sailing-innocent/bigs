#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.lerf_utils import smooth, vis_mask_save, polygon_to_mask, stack_mask, show_result

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def main():
    print("Evaluate LERF")
    lerf_path = "data/datasets/lerf_ovs/label"
    output_path = "data/mid/bigs/lerf_eval_results"
    name = "teatime"
    lerf_path = os.path.join(lerf_path, name)
    output_path = os.path.join(output_path, name)

    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path)
    
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(lerf_path, output_path)
    frame = '1'
    query = 'stuffed bear'


    print("GT annotations organized")
    print(gt_ann.keys())
    print(gt_ann[frame].keys())
    print(gt_ann[frame][query].keys())
    
    mask = gt_ann[frame][query]['mask']
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

    print("image shape: ", image_shape)
    print("image paths: ", image_paths)

main()