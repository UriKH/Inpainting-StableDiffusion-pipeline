from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Any, Callable
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
import os
import torch
import json
from tqdm import tqdm

class InpaintPipelineInput:
    # Use NEAREST to keep masks strictly binary, and target 512x512
    MASK_DEFAULT_PREPROC_OP = staticmethod(lambda mask_image: mask_image.resize((512, 512), Image.NEAREST))

    def __init__(self, prompt, init_image, mask_image, mask_op=None):
        self.prompt = prompt
        self.init_image = init_image
        self.mask_image = mask_image

        if mask_op is None:
            mask_op = self.__class__.MASK_DEFAULT_PREPROC_OP

        if isinstance(self.init_image, str):
            self.init_image = Image.open(self.init_image)
            
        # Warning: Direct resize stretches non-square images! (See COCO practices below)
        self.init_image = self.init_image.convert("RGB").resize((512, 512), Image.LANCZOS)

        if isinstance(self.mask_image, str):
            self.mask_image = Image.open(self.mask_image)
            
        self.mask_image = self.mask_image.convert("L")
        self.mask_image = mask_op(self.mask_image)



class SD2InpaintingPipeLineScheme(ABC):
    def __init__(self, model_id, device):
        self.device = device
        self.vae, self.unet, self.text_encoder, self.tokenizer, self.scheduler = self.load_sd2_components(model_id,
                                                                                                          device=device)

    @staticmethod
    def load_sd2_components(model_path, device="cuda"):
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float32).to(device)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                     torch_dtype=torch.float32).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch.float32)
        print('Components loaded successfully')
        return vae, unet, text_encoder, tokenizer, scheduler

    @abstractmethod
    def pipe(self, pipe_in: InpaintPipelineInput):
        raise NotImplementedError

    def apply_multiple(self, dir_in: str, dir_out: str, is_coco: bool = False, num_coco: int = 1000):
        os.makedirs(dir_out, exist_ok=True)
        if not is_coco:
            file_names = {file.split('.')[0] for file in os.listdir(dir_in)}

            for name in tqdm(file_names, desc='Apply pipeline: '):
                if len(name) == 0:
                    continue
                prompt = None
                with open(os.path.join(dir_in, f'{name}.txt'), "r") as f:
                    prompt = f.read().strip()

                mask = Image.open(os.path.join(dir_in, f'{name}.mask.png')).convert('RGB')
                image = Image.open(os.path.join(dir_in, f'{name}.png')).convert('RGB')
                pipe_in = InpaintPipelineInput(prompt, image, mask)
                result = self.pipe(pipe_in)
                result.save(os.path.join(dir_out, f'{name}.png'))
        
        else:
            base_dir = os.path.dirname(dir_in)
            split = 'val2017'
            captions_file = os.path.join(base_dir, f'annotations/captions_{split}.json')
            instances_file = os.path.join(base_dir, f'annotations/instances_{split}.json')
            
            print("Loading COCO annotations...")
            with open(captions_file, 'r') as f:
                captions_data = json.load(f)
            with open(instances_file, 'r') as f:
                instances_data = json.load(f)

            captions_dict = {}
            for ann in captions_data['annotations']:
                img_id = ann['image_id']
                if img_id not in captions_dict:
                    captions_dict[img_id] = ann['caption']

            instances_dict = {}
            for ann in instances_data['annotations']:
                img_id = ann['image_id']
                if img_id not in instances_dict:
                    instances_dict[img_id] = []
                instances_dict[img_id].append(ann)

            images = [f for f in os.listdir(dir_in) if f.endswith('.jpg')][:num_coco]

            for filename in tqdm(images, desc='Apply pipeline to COCO: '):
                image_id = int(filename.split('.')[0])
                prompt = captions_dict.get(image_id, "a photo of an object")
                
                img_path = os.path.join(dir_in, filename)
                init_image = Image.open(img_path).convert("RGB")
                orig_width, orig_height = init_image.size
                
                mask_image = Image.new("RGB", (orig_width, orig_height), (0, 0, 0))
                draw = ImageDraw.Draw(mask_image)
                
                anns = instances_dict.get(image_id, [])
                if anns:
                    bbox = anns[0]['bbox']
                    x, y, w, h = bbox
                    draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255))
                else:
                    draw.rectangle((orig_width//4, orig_height//4, orig_width*3//4, orig_height*3//4), fill=(255, 255, 255))
                
                pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
                result = self.pipe(pipe_in)
                out_filename = filename.replace('.jpg', '.png')
                result.save(os.path.join(dir_out, out_filename))
