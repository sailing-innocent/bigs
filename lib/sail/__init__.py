__all__ = ["sail", "__version__", "__doc__", "__name__", "add"]

from .module import SailModule

sail = SailModule.get_instance()
# Attributes
# __version__ = sailtorch["__version__"]
# __doc__ = sailtorch["__doc__"]
__name__ = "sail"
# Functions
add = sail("add")
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def point_vis_impl(
    d_pos,
    d_color,
    num_points,
    debug_lines=[],
    point_size=10.0,
    width=800,
    height=600,
    d_pos_stride=3,
    d_color_stride=3,
):
    func = sail("point_vis")
    args = (
        d_pos,
        d_color,
        num_points,
        debug_lines,
        point_size,
        width,
        height,
        d_pos_stride,
        d_color_stride,
    )
    return func(*args)


def point_vis(
    d_pos: torch.Tensor,
    d_color: torch.Tensor,
    debug_lines=[],
    point_size=10.0,
    width=800,
    height=600,
    d_pos_stride=3,
    d_color_stride=3,
):
    # Check the size and device of pos and color
    if not isinstance(d_pos, torch.Tensor) or not isinstance(d_color, torch.Tensor):
        raise TypeError("d_pos and d_color must be torch tensors.")
    num_points = d_pos.size(0)
    if d_pos.size(0) != d_color.size(0):
        logger.info(f"d_pos size: {d_pos.size(0)}, d_color size: {d_color.size(0)}")
        raise ValueError(
            "d_pos and d_color must have the same number of points as num_points."
        )
    if d_pos.device != d_color.device:
        raise ValueError("d_pos and d_color must be on the same device.")

    # impl
    point_vis_impl(
        d_pos.contiguous().data_ptr(),
        d_color.contiguous().data_ptr(),
        num_points,
        debug_lines,
        point_size,
        width,
        height,
        d_pos_stride,
        d_color_stride,
    )
