# -*- coding: utf-8 -*-
# @file blender_utils.py
# @brief Blender Utilities, for debugging Camera
# @author sailing-innocent
# @date 2025-02-24
# @version 1.0
# ---------------------------------

import numpy as np 
import bpy 
from mathutils import Vector
from functools import wraps
import os 

def bopen(mainfile_path='sample.blend', clear=True):
    # INIT SAVE
    if (not os.path.exists(mainfile_path)):
        bpy.ops.wm.save_mainfile(filepath=mainfile_path)

    bpy.ops.wm.open_mainfile(filepath = mainfile_path)
    if (clear):
        bclear()

def bclose(mainfile_path='sample.blend'):
    bpy.ops.wm.save_mainfile(filepath=mainfile_path)

def bclear():
    # remove all elements
    # object mode
    # test if there is any object
    bpy.context.scene.cursor.location = (0, 0, 0)
    if (len(bpy.data.objects) == 0):
        return

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def blender_executive(func):
    @wraps(func)
    def wrapper(
        filename: str = "filename",
        subfolder: str = "test", 
        rootdir: str = "data/mid/blender",
        clear: bool = True,
        **kwargs
    ):
        cwd = os.path.join(rootdir, subfolder)
        if not os.path.exists(cwd):
            os.makedirs(cwd)
        name = filename
        mainfile_path = os.path.join(cwd, name + ".blend")
        bopen(mainfile_path, clear=clear)
        # core func
        func(rootdir, **kwargs)
        # save and close
        bclose(mainfile_path)
        
    return wrapper

def create_basic_camera(origin=Vector((0.0, 0.0, 2.0)), lens=25, clip_start=0.1, clip_end=100, camera_type='PERSP', ortho_scale=6, name="Camera"):
    # Create object and camera
    camera = bpy.data.cameras.new(name)
    camera.lens = lens 
    camera.clip_start = clip_start 
    camera.clip_end = clip_end 
    camera.type = camera_type # 'PERSP', 'ORTHO', 'PANO'

    if (camera_type == 'ORTHO'):
        camera.ortho_scale = ortho_scale
    # Link Object to Scene 
    obj = bpy.data.objects.new("CameraObj", camera)
    # Our Camera System is different from Blender's Default Settings 
    obj.location = origin 
    bpy.context.collection.objects.link(obj)
    bpy.context.scene.camera = obj # Make Current
    return obj 

