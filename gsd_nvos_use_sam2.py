import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse 
from config import get_nvos_json
from utils.image_utils import show_mask, show_points

device = torch.device("cuda")
from sam2.build_sam import build_sam2_video_predictor # type: ignore

def sam_seg(scene: str, debug: bool):
    config = get_nvos_json()
    if scene not in config:
        raise ValueError(f"scene {scene} not found in nvos.json")
    scene_config = config[scene]
    video_dir = f"data/mid/gsd/{scene}/frames"
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0

    sample_img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    w = sample_img.width
    h = sample_img.height
    print(f"frame size: {w}x{h}")
    # sam2_checkpoint = "data/pretrained/sam2/sam2.1_hiera_tiny.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint = "data/pretrained/vision/sam2/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    annotations = scene_config["annotations"]
    for ann in annotations:
        ann_frame_idx = ann["frame_idx"]
        ann_obj_id = 1 # simple one object case
        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = ann["points"]
        labels = ann["labels"]
        points = np.array(points, dtype=np.float32)
        points = points * np.array([w, h], dtype=np.float32)
        N_points = len(points)
        labels = np.array(labels, dtype=np.int64)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        # show the results on the current (interacted) frame
        if (debug):
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            print(frame_names[ann_frame_idx])
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            plt.show()

    if (debug):
        # stop here if debug mode is on
        return
    
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    mask_out_dir = os.path.join(video_dir, "masks")
    os.makedirs(mask_out_dir, exist_ok=True)
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")  
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            # save mask to video_dir/masks
            h, w = out_mask.shape[-2:]
            mask_image = out_mask.reshape(h, w)
            # convert False to 0 and True to 255
            mask_image = mask_image.astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_image, "L")
            mask_image.save(os.path.join(mask_out_dir, f"mask_{out_frame_idx}_{out_obj_id}.png"))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSD Object Removal")
    parser.add_argument("--scene", type=str, default="horns_center")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    scene = args.scene
    sam_seg(scene, args.debug)