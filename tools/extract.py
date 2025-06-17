# -*- coding: utf-8 -*-
# @file extract.py
# @brief Extract the toolkits
# @author sailing-innocent
# @date 2025-02-25
# @version 1.0
# ---------------------------------

__all__ = [
    "toolkits"
]

from plyfile import PlyData, PlyElement
import numpy as np 

class ExtractXYZParams:
    def __init__(self, args):
        self.param_group_name = "ExtractXYZParams"
        try:
            self.input = args.input 
            self.output = args.output
        except:
            print(f"Invalid arguments for {self.param_group_name}")
            exit(1)
    def __str__(self):
        return f"Input: {self.input}, Output: {self.output}"

def extract_xyz(args):
    params = ExtractXYZParams(args)
    print(params)
    input_f = params.input 
    output_f = params.output    

    try:
        plydata = PlyData.read(input_f)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_f)
    except:
        print(f"Error parsing {input_f} and writing {output_f}")
        exit(1)

toolkits = {
    "extract_xyz": extract_xyz
}