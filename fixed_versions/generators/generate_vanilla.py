import os

from vanilla_pipeline import InpaintPipeline
import torch

import argparse
import torch_utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation tool")
    parser.add_argument("path_in", help="Path to the input images.")
    parser.add_argument("path_out", help="Path to the output images.")

    args = parser.parse_args()
    name = args.path.split(os.path.sep)[-1]

    utils.print_title('Generate using Vanilla Pipeline')
    p = InpaintPipeline()
    p.apply_multiple(args.path_in, args.path_out)
