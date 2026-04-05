from abc import ABC, abstractmethod
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
import numpy as np


class InpaintPipelineInput:
    """
    Inpaint pipeline input structure
    """

    def __init__(self, prompt: str, init_image: Image.Image, mask_image: Image.Image):
        """
        :param prompt: The text prompt to guide the image generation
        :param init_image: The image to be inpainted
        :param mask_image: The mask image to guide the inpainting process
        """
        self.prompt = prompt
        self.init_image = init_image
        self.mask_image = mask_image

        if isinstance(self.init_image, str):
            self.init_image = Image.open(self.init_image)
            
        self.init_image = self.init_image.convert("RGB")

        if isinstance(self.mask_image, str):
            self.mask_image = Image.open(self.mask_image)
            
        self.mask_image = self.mask_image.convert("L")

        # masking to ensure no information leakage
        img_arr = np.array(self.init_image)
        mask_arr = np.array(self.mask_image)
        img_arr[mask_arr == 255] = 0
        self.init_image = Image.fromarray(img_arr)


class InpaintingPipeLineScheme(ABC):
    def __init__(self, model_id, device, reconstruction=False, init_noise_strength=1.0, **kwargs):
        """
        :param model_id: The HG path to the Stable Diffusion 2 base model.
        :param device: The device to use for inference ('cuda' or 'cpu').
        :param reconstruction: Whether to use reconstruction methods or filling.
        :param init_noise_strength: The strength of the initial noise added to the image.
        """
        self.device = device
        self.reconstruction = reconstruction
        self.init_noise_strength = init_noise_strength
        self.vae, self.unet, self.text_encoder, self.tokenizer, self.scheduler = self.load_sd2_components(model_id, device=device)

    @staticmethod
    def load_sd2_components(model_path, device="cuda"):
        """
        Loads all relevant components for using Stable Diffusion 2 base model.
        """
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
        :param init_image: The image to be inpainted.
        :param mask_image: The mask image to guide the inpainting process.
        :param target_size: The base size the image will be resized to.
        """
        original_width, original_height = init_image.size
        scale = target_size / max(original_width, original_height)
        w = int(original_width * scale)
        h = int(original_height * scale)
        img_resized = init_image.resize((w, h), Image.LANCZOS)
        mask_resized = mask_image.resize((w, h), Image.NEAREST)

        pad_w = target_size - w
        pad_h = target_size - h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        img_arr = np.array(img_resized)
        mask_arr = np.array(mask_resized)
        img_padded_arr = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
        mask_padded_arr = np.pad(mask_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                                 constant_values=0)
        return Image.fromarray(img_padded_arr), Image.fromarray(mask_padded_arr)

    @staticmethod
    def restore_original_dimensions(
            generated_image: Image.Image,
            original_size: tuple[int, int],
            target_size=512) -> Image.Image:
        """
        Removes the padding from the pipeline's output and resizes it back to the original image's dimensions.
        :param generated_image: The image output by the Stable Diffusion pipeline.
        :param original_size: A tuple of (original_width, original_height).
        :param target_size: The base size the image was padded to.
        """
        original_width, original_height = original_size
        scale = target_size / max(original_width, original_height)
        w = int(original_width * scale)
        h = int(original_height * scale)
        pad_w = target_size - w
        pad_left = pad_w // 2
        pad_h = target_size - h
        pad_top = pad_h // 2

        cropped_image = generated_image.crop((pad_left, pad_top, pad_left + w, pad_top + h))
        final_image = cropped_image.resize((original_width, original_height), Image.LANCZOS)
        return final_image

    @abstractmethod
    def pipe(self, pipe_in: InpaintPipelineInput,  target_size=512):
        raise NotImplementedError
