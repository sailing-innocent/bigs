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


def point_vis(
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
