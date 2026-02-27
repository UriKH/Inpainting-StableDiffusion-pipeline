from vanilla_pipeline import InpaintPipeline as VanillaPipeLine
from ready_pipeline import PipeLine as ReadyPipeLine
from improved_pipeline import ImprovedInpaintPipeline as ImprovedPipeLine
import torch

import argparse


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation tool")
    parser.add_argument("-v", "--vanilla", default=".", help="Folder path to save the resulting images.")
    parser.add_argument("-b", "--basic", default=".", help="Folder path to save the resulting images.")
    parser.add_argument("-i", "--improved", default=".", help="Folder path to save the resulting images.")

    args = parser.parse_args()

    if args.vanilla:
        p = VanillaPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_vanilla')
    
        del p
        clear_cache()

    if args.basic:
        p = ReadyPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_ready_made')

        del p
        clear_cache()

    if args.improved:
        p = ImprovedPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_improved')

