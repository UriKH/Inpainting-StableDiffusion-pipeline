import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vanilla_pipeline import InpaintPipeline
import torch

import argparse
from utils.printing import print_title

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation tool")
    parser.add_argument("path_in", help="Path to the input images.")
    parser.add_argument("path_out", help="Path to the output images.")

    args = parser.parse_args()
    name = args.path_in.split(os.path.sep)[-1]

    print_title('Generate using Vanilla Pipeline')
    p = InpaintPipeline()
    p.apply_multiple(args.path_in, args.path_out)
