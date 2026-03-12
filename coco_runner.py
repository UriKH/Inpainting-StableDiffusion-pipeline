import os
import sys
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline import InpaintPipelineInput, InpaintingPipeLineScheme
from mask_generator import MaskGenerator
from utils.globals import MASKING_CONFIGS
from utils.seed import seed_everything

base_seed = 42


class COCODatasetGenerator:
    """
    A class for generating images from COCO dataset using a specific pipeline.
    """

    def __init__(self, instances_json_path: str, captions_json_path: str):
        """
        :param instances_json_path: Path to the COCO instances JSON file.
        :param captions_json_path: Path to the COCO captions JSON file.
        """
        print('====== loading COCO ======')
        self.img_filename_to_id, self.img_id_to_caption = self.__load_coco_data(instances_json_path, captions_json_path)
        self.mask_generator = MaskGenerator(**MASKING_CONFIGS)

    @staticmethod
    def __load_coco_data(instances_json_path: str, captions_json_path: str):
        """
        load bounding boxes and captions from COCO dataset
        :param instances_json_path: Path to the COCO instances JSON file.
        :param captions_json_path: Path to the COCO captions JSON file.
        """
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f)

        img_filename_to_id = {img['file_name']: img['id'] for img in instances_data['images']}

        with open(captions_json_path, 'r') as f:
            captions_data = json.load(f)

        img_id_to_caption = {}
        for cap in tqdm(captions_data['annotations'], desc='loading COCO captions...'):
            if cap['image_id'] not in img_id_to_caption:
                img_id_to_caption[cap['image_id']] = cap['caption'].strip().rstrip('.')
        return img_filename_to_id, img_id_to_caption

    def generate(self, input_path: str, output_dir: str, pipeline: InpaintingPipeLineScheme) -> None:
        """
        Generate images from COCO dataset using a specific pipeline.
        :param input_path: Path to the input directory containing images.
        :param output_dir: Path to the output directory where generated images will be saved.
        :param pipeline: The pipeline to use for generating images.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i, filename in tqdm(enumerate(image_files), desc="Processing COCO images"):
            img_path = os.path.join(input_path, filename)
            init_image = Image.open(img_path).convert("RGB")
            try:
                prompt, img_id = self.get_prompt_img_id(img_path)
            except Exception as e:
                print(f'unexpected error: {e} (continue anyway!)')
                continue
            
            seed_everything(base_seed + img_id)
            mask_image, coverage_ratio = self.mask_generator(np.array(init_image), img_id)
            mask_image = Image.fromarray(mask_image).convert("L")
            pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
            result_img = pipeline.pipe(pipe_in)
            out_path = os.path.join(output_dir, filename)
            result_img.save(out_path)

    def get_prompt_img_id(self, image_path: str) -> Tuple[str, int]:
        """
        Get the prompt and image ID for a given image path.
        :param image_path: Path to the image.
        """
        img_id = self.get_img_id(image_path)
        if img_id not in self.img_id_to_caption:
            raise ValueError(f"No annotations found for image {image_path}.")
        global_caption = self.img_id_to_caption[img_id]
        return global_caption, img_id

    def get_img_id(self, image_path: str) -> int:
        """
        Get the image ID for a given image path.
        :param image_path: Path to the image.
        """
        filename = os.path.basename(image_path)
        if filename not in self.img_filename_to_id:
            raise ValueError(f"Image {image_path} not found in the dataset.")
        return self.img_filename_to_id[filename]
