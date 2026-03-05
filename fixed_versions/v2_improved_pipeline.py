import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2 as cv
from v1_improved_pipeline import ImprovedInpaintPipeline as V1Pipeline
from pipeline import InpaintPipelineInput
from utils import torch_utils as utils

class ImprovedInpaintPipeline(V1Pipeline):
    def __init__(self, jump_length=10, jump_n_sample=2):
        # Inherit Time-Travel parameters and Negative Prompts from V1
        super().__init__(jump_length=jump_length, jump_n_sample=jump_n_sample)
        self.feather_radius = 10
        self.dilate_kernel_size = 5

    def preprocess(self, pipe_in: InpaintPipelineInput):
        """
        Enhances the mask using Dilation and Feathering to prevent 'cut' edges.
        """
        # 1. Dilate mask: Expand slightly to give U-Net context inside the boundary
        mask_np = np.array(pipe_in.mask_image.convert("L"))
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        
        # 2. Feather mask: Use Gaussian blur to create a soft transition gradient
        mask_pil = Image.fromarray(mask_dilated).filter(ImageFilter.GaussianBlur(radius=self.feather_radius))
        pipe_in.mask_image = mask_pil

        # 3. Smart Crop Logic (Implicit): 
        # By utilizing the base resize_pipe which maintains aspect ratio via padding,
        # the model avoids the distortion that causes 'half-person' artifacts.
        return super().preprocess(pipe_in)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        """
        Combines Time-Travel Resampling (V1) with Annealed Blending (V2).
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        schedule_indices = self._get_repaint_schedule(num_inference_steps) # From V1
        
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))

        for idx, step_index in enumerate(schedule_indices):
            t = self.scheduler.timesteps[step_index]
            
            # Classifier Free Guidance logic from V1
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # --- ANNEALED BLENDING LOGIC ---
            # Gradually reduce mask influence as t -> 0 to allow texture continuity.
            # Progresses from 0.0 (start) to 1.0 (end)
            progress = idx / len(schedule_indices)
            anneal_factor = 1.0 - (0.15 * progress) # Retain 85% of background strength at the end
            current_mask = mask_tensor * anneal_factor

            # Handle Time-Travel (V1 logic) and Background Context
            is_last_step = (idx == len(schedule_indices) - 1)
            if not is_last_step:
                next_step_index = schedule_indices[idx + 1]
                t_next = self.scheduler.timesteps[next_step_index]
                
                if next_step_index < step_index: # Backward jump
                    beta = self.scheduler.betas[t_next].to(self.device)
                    latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * torch.randn_like(latents)
                
                background_noise = torch.randn_like(init_latents)
                known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
            else:
                known_background = init_latents

            # Blend with Annealed/Feathered Mask
            latents = (known_background * (1 - current_mask)) + (latents * current_mask)
            
        return latents
