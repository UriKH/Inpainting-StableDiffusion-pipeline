import os
import sys
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vanilla_pipeline import InpaintPipeline
from pipeline import InpaintPipelineInput


class COCODatasetGenerator:
    def __init__(self, instances_json_path, captions_json_path):
        print('=== lodaing COCO ===')
        self.img_filename_to_id, self.img_id_to_ann, self.cat_id_to_name, self.img_id_to_caption = self.__load_coco_data(instances_json_path, captions_json_path)

    @staticmethod
    def __load_coco_data(instances_json_path, captions_json_path):
        """
        load bounding boxes and captions from COCO dataset
        """
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f)

        img_filename_to_id = {img['file_name']: img['id'] for img in instances_data['images']}
        cat_id_to_name = {cat['id']: cat['name'] for cat in instances_data['categories']}

        img_id_to_ann = {}
        for ann in tqdm(instances_data['annotations'], desc='loading COCO annotations...'):
            if ann['image_id'] not in img_id_to_ann:
                img_id_to_ann[ann['image_id']] = ann

        with open(captions_json_path, 'r') as f:
            captions_data = json.load(f)

        img_id_to_caption = {}
        for cap in tqdm(captions_data['annotations'], desc='loading COCO captions...'):
            if cap['image_id'] not in img_id_to_caption:
                img_id_to_caption[cap['image_id']] = cap['caption'].strip().rstrip('.')
        return img_filename_to_id, img_id_to_ann, cat_id_to_name, img_id_to_caption

    def generate(self, input_path, output_dir, pipeline):
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in tqdm(image_files, desc="Processing COCO validation images"):
            img_path = os.path.join(input_path, filename)
            init_image = Image.open(img_path).convert("RGB")
            try:
                prompt, bbox = self.get_mask_prompt(img_path)
            except Exception as e:
                print(f'unexpected error: {e} (continue anyway!)')
                continue
            mask_image = Image.new("L", init_image.size, 0)
            draw = ImageDraw.Draw(mask_image)
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], fill=255)

            pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
            result_img = pipeline.resize_pipe(pipe_in)

            out_path = os.path.join(output_dir, filename)
            result_img.save(out_path)

    def get_mask_prompt(self, image_path) -> Tuple[str, Image.Image]:
        filename = os.path.basename(image_path)
        if filename not in self.img_filename_to_id:
            raise ValueError(f"Image {image_path} not found in the dataset.")

        img_id = self.img_filename_to_id[filename]
        if img_id not in self.img_id_to_ann or img_id not in self.img_id_to_caption:
            raise ValueError(f"No annotations found for image {image_path}.")

        # 1. חילוץ המידע מה-JSON
        ann = self.img_id_to_ann[img_id]
        bbox = ann['bbox']
        category_name = self.cat_id_to_name[ann['category_id']]
        global_caption = self.img_id_to_caption[img_id]

        # 2. יצירת הפרומפט החדש בתבנית המבוקשת
        prompt = f"{category_name}, perfectly integrated into a scene of {global_caption}"
        return prompt, bbox
