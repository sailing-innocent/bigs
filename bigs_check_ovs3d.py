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

def eval_gt_ovsdata(seg_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise 's gt annotations
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    class_file = os.path.join(seg_folder, 'class.txt')
    with open(class_file, 'r') as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes]
    print("classes: ", classes)



def main():
    print("Evaluate 3D OVS data")
    lerf_path = "data/datasets/ovs3d/"
    output_path = "data/mid/bigs/ovs_eval_results"
    name = "teatime"
    lerf_path = os.path.join(lerf_path, name)
    output_path = os.path.join(output_path, name)

    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path)
    
    gt_ann, image_shape, image_paths = eval_gt_ovsdata(lerf_path, output_path)
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