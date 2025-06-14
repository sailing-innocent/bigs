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
    name="bigs",
    packages=["bigs"],
    ext_modules=[
        CUDAExtension(
            name="bigs._C",
            sources=[
                "gs_state.cu",
                "featmark.cu",
                "featmark_impl.cu",
                "featmark_point.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "../third_party/"
                    )
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
