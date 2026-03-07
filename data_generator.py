import importlib
import argparse
import traceback
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coco_runner import COCODatasetGenerator
from utils.getters import input_output_paths_args
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH


def generate(input_paths, output_paths, pipeline):
    generator = COCODatasetGenerator(
        COCO_INSTANCES_PATH,
        COCO_CAPTIONS_PATH,
    )
    generator.generate(input_paths, output_paths, pipeline)

def main():
    parser = argparse.ArgumentParser(description="which pipeline version to import")
    
    parser.add_argument("--sm_dilation_kernel", default=5, type=int, help="dilation kernel for soft masking")
    parser.add_argument("--sm_blur_kernel", default=15, type=int, help="bluring kernel for soft masking")
    parser.add_argument("--sm_sigma", default=5.0, type=float, help="sigma for soft masking")
    parser.add_argument("--rp_jump_length", default=10, type=int, help="jump length for RePaint")
    parser.add_argument("--rp_jump_n_sample", default=2, type=int, help="number of jumps for RePaint")

    parser.add_argument("--version", type=int, required=True, help="insert the version number (0 for vanilla)")

    input_paths, output_paths = input_output_paths_args(parser)
    args = parser.parse_args()
    
    i = int(args.version)
    if i == 0:
        module_name = 'pipelines.vanilla_pipeline'
        class_name = 'InpaintPipelineVanilla'
    else:
        module_name = f'pipelines.v{i}_improved' 
        class_name = f'ImprovedInpaintPipelineV{i}'
        
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            print(f"Error: Module '{module_name}' not found.")
        else:
            print(f"Error while importing '{module_name}': missing dependency '{e.name}'.")
            traceback.print_exc()
        return
    except ImportError as e:
        print(f"ImportError while loading '{module_name}': {e}")
        traceback.print_exc()
        return

    try:
        target_class = getattr(module, class_name)
        params = vars(args)
        params.pop('version')
        instance = target_class(**params)
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in '{module_name}'.")
        return

    try:
        generate(input_paths, output_paths, instance)
    except Exception as e:
        print(f"Error while running generator: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
