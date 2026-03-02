import os

from vanilla_pipeline import InpaintPipeline as VanillaPipeLine
from ready_pipeline import PipeLine as ReadyPipeLine
from improved_pipeline_v7 import ImprovedInpaintPipeline as ImprovedPipeLine
import torch

import argparse
import torch_utils as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation tool")
    parser.add_argument("-v", "--vanilla", action='store_true', help="Folder path to save the resulting images.")
    parser.add_argument("-b", "--basic", action='store_true', help="Folder path to save the resulting images.")
    parser.add_argument("-i", "--improved", action='store_true', help="Folder path to save the resulting images.")
    parser.add_argument("-c", "--coco", action='store_true', help="Indicate if the dataset is COCO.")
    parser.add_argument("-n", "--num", type=int, default=1000, help="Number of COCO images to process.")
    parser.add_argument("--blur", default='1')
    parser.add_argument("--dilate", default='1')
    parser.add_argument("--resample", default='1')
    parser.add_argument("path", help="Path to the input images.")

    args = parser.parse_args()
    name = args.path.split(os.path.sep)[-1]
    dataset_type = "coco" if args.coco else "ours"

    utils.clear_cache()

    if args.vanilla:
        utils.print_title('Generate using Vanilla Pipeline')
        p = VanillaPipeLine()
        p.apply_multiple(args.path, rf'./data/{dataset_type}/{name}_results_vanilla', is_coco=args.coco, num_coco=args.num)
    
        del p
        utils.clear_cache()
        
    if args.basic:
        utils.print_title('Generate using Ready-Made Pipeline')
        p = ReadyPipeLine()
        p.apply_multiple(args.path, rf'./data/{dataset_type}/{name}_results_ready_made', is_coco=args.coco, num_coco=args.num)

        del p
        utils.clear_cache()

    if args.improved:
        utils.print_title('Generate using Improved Pipeline')
        p = ImprovedPipeLine() #(int(args.blur), int(args.dilate), int(args.resample))
        p.apply_multiple(args.path, rf'./data/{dataset_type}/{name}_results_improved', is_coco=args.coco, num_coco=args.num)
    
        del p
        utils.clear_cache()
