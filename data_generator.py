import importlib
import argparse
import sys

from generators.coco_runner import COCODatasetGenerator
from utils.getters import input_output_paths_args
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH


def generate(input_paths, output_paths, pipeline_cls):
    pipeline = pipeline_cls()
    generator = COCODatasetGenerator(
        COCO_INSTANCES_PATH,
        COCO_CAPTIONS_PATH
    )
    generator.generate(input_paths, output_paths, pipeline)

def main():
    parser = argparse.ArgumentParser(description="which pipeline version to import")
    parser.add_argument("--version", type=int, required=True, help="insert the version number (0 for vanilla)")
    input_paths, output_paths = input_output_paths_args(parser)
    args = parser.parse_args()

    i = int(args.version)
    if i == 0:
        module_name = 'vanilla_pipeline'
        class_name = 'InpaintPipelineVanilla'
    else:
        module_name = f'v{i}_improved'
        class_name = f'ImprovedInpaintPipelineV{i}'

    try:
        module = importlib.import_module(module_name)
        target_class = getattr(module, class_name)
        generate(input_paths, output_paths, target_class)
    except ImportError:
        print(f"Error: Module '{module_name}' not found.")
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in '{module_name}'.")

if __name__ == "__main__":
    main()