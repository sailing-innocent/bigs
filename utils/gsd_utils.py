from scene.cameras import get_round_cam
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos
import numpy as np


def get_cam_list_round(cam_config: dict, world_name: str, N_frames: int):
    cam_list = []
    rot_center_point = cam_config["rot_center_point"]
    eye_fix_point = cam_config["eye_fix_point"]
    rot_axis = cam_config["rot_axis"]
    scale = cam_config["scale"]
    start_theta = cam_config["start_theta"]
    v_width = cam_config["v_width"]
    v_height = cam_config["v_height"]

    for i in range(N_frames):
        theta = start_theta + i * 2 * np.pi / N_frames
        cam, _ = get_round_cam(
            rot_center_point,
            rot_axis,
            eye_fix_point,
            theta,
            scale=scale,
            world_name=world_name,
            v_width=v_width,
            v_height=v_height,
        )
        cam_list.append(cam)
    return cam_list


def get_cam_list_dataset(dataset_path: str, world_name: str = "Colmap"):
    cam_list = []
    scene_info = readColmapSceneInfo(dataset_path, None, False)
    cam_list = cameraList_from_camInfos(scene_info.train_cameras, 1, -1)
    return cam_list
