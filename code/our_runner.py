import os
import sys
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline import InpaintPipelineInput, InpaintingPipeLineScheme
from mask_generator import MaskGenerator
from utils.globals import MASKING_CONFIGS
from utils.seed import set_seed

base_seed = 42


class OurDatasetGenerator:
    """
    A class for generating images from our dataset using a specific pipeline.
    """

    def __init__(self, instance_dir: str, captions_json_path: str):
        """
        :param instance_dir: Path to the instances directory containing images.
        :param captions_json_path: Path to the captions JSON file.
        """
        print('====== loading our dataset ======')
        self.img_filename_to_id, self.img_id_to_caption = self.__load_our_dataset(instance_dir, captions_json_path)
        self.mask_generator = MaskGenerator(**MASKING_CONFIGS)

    @staticmethod
    def __load_our_dataset(instance_dir: str, captions_json_path: str):
        """
        load bounding boxes and captions from dataset
        :param instance_dir: Path to the instances directory containing images.
        :param captions_json_path: Path to the captions JSON file.
        """
        with open(captions_json_path, 'r') as f:
            prompts = json.load(f)
            prompts = {int(k): v for k, v in prompts.items()}

        f_names = os.listdir(instance_dir)
        img_filename_to_id = {name: int(name.split('.')[0]) for name in f_names if name.lower().endswith(('.png', '.jpg', '.jpeg'))}
        return img_filename_to_id, prompts

    def generate(self, input_path: str, output_dir: str, pipeline: InpaintingPipeLineScheme) -> None:
        """
        Generate images from our dataset using a specific pipeline.
        :param input_path: Path to the input directory containing images.
        :param output_dir: Path to the output directory where generated images will be saved.
        :param pipeline: The pipeline to use for generating images.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i, filename in tqdm(enumerate(image_files), desc="Processing our images"):
            img_path = os.path.join(input_path, filename)
            init_image = Image.open(img_path).convert("RGB")
            try:
                img_id = self.img_filename_to_id[filename]
                prompt = self.img_id_to_caption[img_id]
            except Exception as e:
                print(f'unexpected error: {e} (continue anyway!)')
                continue

            set_seed(base_seed + img_id)
            mask_image, coverage_ratio = self.mask_generator(np.array(init_image), img_id)
            mask_image = Image.fromarray(mask_image).convert("L")
            pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)

            # remove this later
            pipe_in.init_image.save(os.path.join(output_dir, f'{img_id}_init.png'))
            result_img = pipeline.pipe(pipe_in)
            out_path = os.path.join(output_dir, filename)
            result_img.save(out_path)
