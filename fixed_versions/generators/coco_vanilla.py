import os
import sys
from coco_runner import COCODatasetGenerator
from utils.getters import input_output_paths_args
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixed_versions.vanilla_pipeline import InpaintPipeline


if __name__ == "__main__":
    input_paths, output_paths = input_output_paths_args()

    pipeline = InpaintPipeline()
    generator = COCODatasetGenerator(
        COCO_INSTANCES_PATH,
        COCO_CAPTIONS_PATH,
        pipeline
    )
    generator.generate(input_paths, output_paths)
