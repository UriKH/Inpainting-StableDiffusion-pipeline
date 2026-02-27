from vanilla_pipeline import InpaintPipeline as VanillaPipeLine
from ready_pipeline import PipeLine as ReadyPipeLine
from improved_pipeline import ImprovedInpaintPipeline as ImprovedPipeLine
import torch

import argparse


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_buffer(msg):
    print(f'{"="* 20} {msg} {"=" * 20}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation tool")
    parser.add_argument("-v", "--vanilla", action='store_true', help="Folder path to save the resulting images.")
    parser.add_argument("-b", "--basic", action='store_true', help="Folder path to save the resulting images.")
    parser.add_argument("-i", "--improved", action='store_true', help="Folder path to save the resulting images.")

    args = parser.parse_args()

    if args.vanilla:
        print_buffer('Generate using Vanilla Pipeline')
        p = VanillaPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_vanilla')
    
        del p
        clear_cache()
        
    if args.basic:
        print_buffer('Generate using Ready-Made Pipeline')
        p = ReadyPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_ready_made')

        del p
        clear_cache()

    if args.improved:
        print_buffer('Geneerate using Improved Pipeline')
        p = ImprovedPipeLine()
        p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_improved')

