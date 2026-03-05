import os
import sys
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vanilla_pipeline import InpaintPipeline
from pipeline import InpaintPipelineInput
from mask_generator import MaskGenerator


class COCODatasetGenerator:
    def __init__(self, instances_json_path, captions_json_path):
        print('====== loading COCO ======')
        self.img_filename_to_id, self.img_id_to_caption = self.__load_coco_data(instances_json_path, captions_json_path)
        self.mask_generator = MaskGenerator(
            max_box_side=64, max_boxes=2, max_strokes=7, max_points=7, min_points=3
        )

    @staticmethod
    def __load_coco_data(instances_json_path, captions_json_path):
        """
        load bounding boxes and captions from COCO dataset
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

    def generate(self, input_path, output_dir, pipeline):
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in tqdm(image_files, desc="Processing COCO validation images"):
            img_path = os.path.join(input_path, filename)
            init_image = Image.open(img_path).convert("RGB")
            try:
                prompt, img_id = self.get_mask_prompt(img_path)
            except Exception as e:
                print(f'unexpected error: {e} (continue anyway!)')
                continue
            mask_image, coverage_ratio = self.mask_generator(np.array(init_image), img_id, 1)
            mask_image = Image.fromarray(mask_image).convert("L")

            pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
            result_img = pipeline.resize_pipe(pipe_in)

            out_path = os.path.join(output_dir, filename)
            result_img.save(out_path)

    def get_mask_prompt(self, image_path) -> Tuple[str, Image.Image]:
        filename = os.path.basename(image_path)
        if filename not in self.img_filename_to_id:
            raise ValueError(f"Image {image_path} not found in the dataset.")

        img_id = self.img_filename_to_id[filename]
        if img_id not in self.img_id_to_caption:
            raise ValueError(f"No annotations found for image {image_path}.")
        global_caption = self.img_id_to_caption[img_id]
        return global_caption, img_id
