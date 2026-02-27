import torch
import numpy as np
import dataclasses
import os

from diffusers import (
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union


@dataclass
class InpaintPipelineInput:
    prompt: str
    init_image: Union[Image.Image, str]
    mask_image: Union[Image.Image, str]

    def __post_init__(self):
        if isinstance(self.init_image, str):
            self.init_image = Image.open(self.init_image).convert("RGB").resize((512, 512))
        if isinstance(self.mask_image, str):
            self.mask_image = Image.open(self.mask_image).convert("L").resize((64, 64))


class SD2InpaintingPipeLineScheme(ABC):
    def __init__(self, model_id, device):
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
        return vae, unet, text_encoder, tokenizer, scheduler

    @abstractmethod
    def pipe(self, pipe_in: InpaintPipelineInput):
        raise NotImplementedError

    def apply_multiple(self, dir_in: str, dir_out: str):
        file_names = {file.split('.')[0] for file in os.listdir(dir_in)}
        os.makedirs(dir_out, exist_ok=True)

        for name in file_names:
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


class InpaintPipeline(SD2InpaintingPipeLineScheme):
    MODEL_ID = "Manojb/stable-diffusion-2-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SD_SCALE_FACTOR = 0.18215

    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        super().__init__(model_id, device)

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def prepare_latents(self, image):
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            init_latents = self.vae.encode(image_tensor).latent_dist.sample()
            init_latents = self.SD_SCALE_FACTOR * init_latents
        return init_latents

    def prepare_mask_tensor(self, mask_image):
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask_tensor

    def decode_latents(self, latents) -> Image.Image:
        with torch.no_grad():
            latents = 1 / self.SD_SCALE_FACTOR * latents
            image = self.vae.decode(latents).sample

        # Convert back to a PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn_like(init_latents)

        print("Starting the custom denoising loop...")
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # Step the scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # --- THE VANILLA INPAINTING MAGIC HAPPENS HERE ---
            # 1. Add the correct amount of noise to the original image for this specific timestep
            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t)

            # 2. Force the background to stay true to the original image, while keeping the AI's generation inside the mask
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
        return latents

    def pipe(self, pipe_in: InpaintPipelineInput):
        text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
        latents = self.prepare_latents(pipe_in.init_image)
        mask = self.prepare_mask_tensor(pipe_in.mask_image)
        latents = self.denoise(text_embeddings, latents, mask)
        final_image = self.decode_latents(latents)
        return final_image
