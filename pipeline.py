from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
import os
import torch
from tqdm import tqdm


@dataclass
class InpaintPipelineInput:
    prompt: str
    init_image: Union[Image.Image, str]
    mask_image: Union[Image.Image, str]

    def __post_init__(self):
        if isinstance(self.init_image, str):
            self.init_image = Image.open(self.init_image)
        self.init_image = self.init_image.convert("RGB").resize((512, 512))

        if isinstance(self.mask_image, str):
            self.mask_image = Image.open(self.mask_image)
        self.mask_image = self.mask_image.convert("L").resize((64, 64))


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
        print('Componenets loaded successfully')
        return vae, unet, text_encoder, tokenizer, scheduler

    @abstractmethod
    def pipe(self, pipe_in: InpaintPipelineInput):
        raise NotImplementedError

    def apply_multiple(self, dir_in: str, dir_out: str):
        file_names = {file.split('.')[0] for file in os.listdir(dir_in)}
        os.makedirs(dir_out, exist_ok=True)

        for name in tqdm(file_names, desc='Apply pipeline: '):
            if len(name) == 0:
                continue
            prompt = None
            with open(os.path.join(dir_in, f'{name}.txt'), "r") as f:
                prompt = f.read()

            mask = Image.open(os.path.join(dir_in, f'{name}.mask.png')).convert('RGB')
            image = Image.open(os.path.join(dir_in, f'{name}.png')).convert('RGB')
            pipe_in = InpaintPipelineInput(prompt, image, mask)
            result = self.pipe(pipe_in)
            result.save(os.path.join(dir_out, f'{name}.png'))
