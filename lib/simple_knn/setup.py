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
import sys


setup(
    name="simple_knn",
    packages=["simple_knn"],
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=["ext.cpp", "simple_knn.cu", "spatial.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
