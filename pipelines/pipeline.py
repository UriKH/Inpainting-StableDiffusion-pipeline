from abc import ABC, abstractmethod
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
import numpy as np


class InpaintPipelineInput:
    def __init__(self, prompt, init_image, mask_image):
        self.prompt = prompt
        self.init_image = init_image
        self.mask_image = mask_image

        if isinstance(self.init_image, str):
            self.init_image = Image.open(self.init_image)
            
        self.init_image = self.init_image.convert("RGB")

        if isinstance(self.mask_image, str):
            self.mask_image = Image.open(self.mask_image)
            
        self.mask_image = self.mask_image.convert("L")

        img_arr = np.array(self.init_image)
        mask_arr = np.array(self.mask_image)
        img_arr[mask_arr == 255] = 0
        self.init_image = Image.fromarray(img_arr)


class InpaintingPipeLineScheme(ABC):
    def __init__(self, model_id, device, **kwargs):
        self.device = device
        self.vae, self.unet, self.text_encoder, self.tokenizer, self.scheduler = self.load_sd2_components(model_id, device=device)

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

    @staticmethod
    def prepare_inpainting_data_advanced(init_image: Image.Image, mask_image: Image.Image, target_size=512):
        """
        Resizes maintaining aspect ratio, edge-pads the image to prevent VAE artifacts,
        and zero-pads the mask to prevent inpainting in the padded regions.
        """
        width, height = init_image.size
        scale = target_size / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)

        # Resize keeping aspect ratio
        img_resized = init_image.resize((new_w, new_h), Image.LANCZOS)
        mask_resized = mask_image.resize((new_w, new_h), Image.NEAREST)

        # Calculate padding for top/bottom and left/right
        pad_w = target_size - new_w
        pad_h = target_size - new_h

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        img_arr = np.array(img_resized)
        mask_arr = np.array(mask_resized)

        # 1. Edge pad the image (repeats the edge pixels)
        # The padding tuple is ((top, bottom), (left, right), (channels))
        img_padded_arr = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')

        # 2. Zero pad the mask (ensures the model ignores the padded area)
        # Mask is 2D (grayscale), so padding tuple is just ((top, bottom), (left, right))
        mask_padded_arr = np.pad(mask_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                 constant_values=0)

        return Image.fromarray(img_padded_arr), Image.fromarray(mask_padded_arr)

    @staticmethod
    def restore_original_dimensions(generated_image: Image.Image, original_size: tuple[int, int],
                                    target_size=512) -> Image.Image:
        """
        Removes the padding from the pipeline's output and resizes it back
        to the original image's dimensions.

        Args:
            generated_image: The 512x512 image output by the Stable Diffusion pipeline.
            original_size: A tuple of (original_width, original_height).
            target_size: The base size the image was padded to (default 512).
        """
        orig_width, orig_height = original_size

        # 1. Recalculate the intermediate scaled dimensions used during padding
        scale = target_size / max(orig_width, orig_height)
        new_w, new_h = int(orig_width * scale), int(orig_height * scale)

        # 2. Recalculate the exact padding offsets
        pad_w = target_size - new_w
        pad_h = target_size - new_h

        pad_left = pad_w // 2
        pad_top = pad_h // 2

        # 3. Crop out the padded edges to retrieve the generated core image
        # The crop box is (left, top, right, bottom)
        right = pad_left + new_w
        bottom = pad_top + new_h
        cropped_image = generated_image.crop((pad_left, pad_top, right, bottom))

        # 4. Resize back to the exact original COCO dimensions
        final_image = cropped_image.resize((orig_width, orig_height), Image.LANCZOS)

        return final_image

    @abstractmethod
    def pipe(self, pipe_in: InpaintPipelineInput):
        raise NotImplementedError

    def resize_pipe(self, pipe_in: InpaintPipelineInput, target_size=512):
        orig_size = pipe_in.init_image.size
        init_image, mask_image = self.prepare_inpainting_data_advanced(pipe_in.init_image, pipe_in.mask_image, target_size)
        pipe_in.init_image = init_image
        pipe_in.mask_image = mask_image
        result_padded = self.pipe(pipe_in)
        return self.restore_original_dimensions(result_padded, orig_size)
