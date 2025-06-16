# -*- coding: utf-8 -*-
# @file camera_utils.py
# @brief camera utils from https://team.inria.fr/graphdeco
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch 

WARNED = False

def loadCamFromInfo(cam_info):
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=cam_info.image, gt_alpha_mask=None,
                  image_name=cam_info.image_name, uid=cam_info.uid, data_device="cuda")

# 什么玩意
def loadCam(downscale_level: int, id, cam_info, resolution_scale, device="cuda"):
    orig_w, orig_h = cam_info.image.size
    if downscale_level in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * downscale_level)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if downscale_level == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / downscale_level

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=device)

def cameraList_from_camInfos(cam_infos, resolution_scale: float = 1.0, downscale_level: int = -1):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(downscale_level, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# from cent.utils.camera import Camera as SailCamera

# def get_round_cam(up, center, theta, scale=3.0, rise_scale=1.0):
#     pos2d = [np.cos(theta), np.sin(theta)]
#     up=np.array(up)
#     pos3d = np.zeros(3)
#     idx = 0
#     for i in range(3):
#         if (up[i] == 0):
#             pos3d[i] = pos2d[idx]
#             idx += 1

#     pos3d = scale * pos3d + center + up * rise_scale
#     sail_cam = SailCamera(flipy=True, up=up)
#     sail_cam.lookat(pos3d, np.array(center))
#     sail_cam.set_res(1000, 500)
#     return sail_cam

