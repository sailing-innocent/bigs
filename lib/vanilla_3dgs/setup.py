# -*- coding: utf-8 -*-
# @file setup.py
# @brief featmark
# @author sailing-innocent
# @date 2025-02-25
# @version 1.0
# ---------------------------------

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="vanilla_3dgs",
    packages=['vanilla_3dgs'],
    ext_modules=[
        CUDAExtension(
            name="vanilla_3dgs._C",
            sources=[
                "rasterize_points.cu",
                "rasterizer_impl.cu",
                "forward.cu",
                "backward.cu",
                "ext.cpp"
            ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "../third_party/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)