# -*- coding: utf-8 -*-
# @file z_tool.py
# @brief Entry point for the toolkit
# @author sailing-innocent
# @date 2025-02-25
# @version 1.0
# ---------------------------------

import argparse 
from tools.extract import toolkits as ExtractToolkits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Toolkit")
    parser.add_argument("--tool", type=str, default="train", help="The tool to use")
    parser.add_argument("--input", type=str, default="input", help="The input file")
    parser.add_argument("--output", type=str, default="output", help="The output file")

    args = parser.parse_args()

    toolkits = {}
    toolkits.update(ExtractToolkits)

    if args.tool in toolkits.keys():
        toolkits[args.tool](args)
    else:
        print("Invalid tool, only the following are supported:")
        print(list(toolkits.keys()))
        exit(1)
