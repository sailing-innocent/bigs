import torch 
from torch.utils.cpp_extension import load 
from . import _C 

def distCUDA2(points: torch.tensor):
    return _C.distCUDA2(points)

