import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
fixed_versions_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(fixed_versions_dir)

sys.path.append(fixed_versions_dir)
sys.path.append(root_dir)

from generators.coco_runner import COCODatasetGenerator
from utils.getters import input_output_paths_args
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH
from v3_improved_pipeline import ImprovedInpaintPipeline


if __name__ == "__main__":
    input_paths, output_paths = input_output_paths_args()

    pipeline = ImprovedInpaintPipeline()
    generator = COCODatasetGenerator(
        COCO_INSTANCES_PATH,
        COCO_CAPTIONS_PATH
    )
    generator.generate(input_paths, output_paths, pipeline)
